//! Chunked deformable heightmap terrain. Heights live in one big vertex grid;
//! chunks are 32x32-cell mesh entities rebuilt when a crater dirties them.
//! Smooth indexed chunks share heightfield normals and blend pasture materials.

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::{Aabb, MeshAabb};
use bevy::image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat};
use bevy::shader::ShaderRef;
use bevy::window::PrimaryWindow;
use std::time::Instant;

use crate::units::hash01;

pub const CELL: f32 = 2.0;
pub const CHUNK_CELLS: usize = 32;
pub const CHUNKS_X: usize = 16;
pub const CHUNKS_Z: usize = 12;
const VERTS_X: usize = CHUNKS_X * CHUNK_CELLS + 1;
const VERTS_Z: usize = CHUNKS_Z * CHUNK_CELLS + 1;

/// The battlefields the menu's Map row cycles through. `FL_MAP=classic`
/// or `FL_MAP=river` picks one at launch; anything else is the grassland.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MapKind {
    /// Broad pasture shoulders around an open lowland: the default.
    #[default]
    Grassland,
    /// The 0.1.0 heightfield: rolling noise with ridged peaks, no river.
    Classic,
    /// Experimental: the same noise with terraces, a river and a bridge.
    River,
}

impl MapKind {
    pub fn from_env() -> Self {
        match std::env::var("FL_MAP").as_deref() {
            Ok("river") => Self::River,
            Ok("classic") => Self::Classic,
            _ => Self::Grassland,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Grassland => "Grassland",
            Self::Classic => "Classic",
            Self::River => "River",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Grassland => Self::Classic,
            Self::Classic => Self::River,
            Self::River => Self::Grassland,
        }
    }
}

/// Sent by the menu when the Map row changes. The terrain regenerates
/// and the river scenery respawns in the same frame.
#[derive(Message, Clone, Copy)]
pub struct MapChanged(pub MapKind);

/// The map rebuild runs in this set; the river scenery respawns after it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct MapRebuild;

#[derive(Resource, Clone)]
pub struct Terrain {
    /// Vertex heights, row-major [z][x].
    heights: Vec<f32>,
    /// Impassable vertices (terrace risers, gorge walls, crater lips),
    /// same grid as `heights`. All-false on the classic map.
    blocked: Vec<bool>,
    pub kind: MapKind,
    /// True on the two maps without a river (grassland and classic):
    /// skips the river carve, terraces, ground variety, and every
    /// river-dependent system (water, bridge, wade).
    pub classic: bool,
    /// World-space min corner.
    pub origin: Vec2,
    dirty: Vec<bool>,
}

impl Terrain {
    pub fn min(&self) -> Vec2 {
        self.origin
    }

    pub fn max(&self) -> Vec2 {
        self.origin + Vec2::new((VERTS_X - 1) as f32 * CELL, (VERTS_Z - 1) as f32 * CELL)
    }

    #[inline]
    fn h(&self, x: usize, z: usize) -> f32 {
        self.heights[z * VERTS_X + x]
    }

    /// Bilinear height sample, clamped to the field. The bridge deck
    /// overrides the carved channel so units walk ON the bridge.
    #[inline]
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        if !self.classic && bridge_deck_contains(x, z) {
            return river_water_level(z) + BRIDGE_DECK_LIFT;
        }
        let g = (Vec2::new(x, z) - self.origin) / CELL;
        let gx = g.x.clamp(0.0, (VERTS_X - 2) as f32);
        let gz = g.y.clamp(0.0, (VERTS_Z - 2) as f32);
        let x0 = gx as usize;
        let z0 = gz as usize;
        let fx = gx - x0 as f32;
        let fz = gz - z0 as f32;
        let h00 = self.h(x0, z0);
        let h10 = self.h(x0 + 1, z0);
        let h01 = self.h(x0, z0 + 1);
        let h11 = self.h(x0 + 1, z0 + 1);
        h00 * (1.0 - fx) * (1.0 - fz)
            + h10 * fx * (1.0 - fz)
            + h01 * (1.0 - fx) * fz
            + h11 * fx * fz
    }

    /// |gradient| of the height field (rise per meter).
    #[inline]
    pub fn slope_at(&self, x: f32, z: f32) -> f32 {
        const E: f32 = 1.0;
        let h = self.height_at(x, z);
        let dx = (self.height_at(x + E, z) - h) / E;
        let dz = (self.height_at(x, z + E) - h) / E;
        (dx * dx + dz * dz).sqrt()
    }

    /// Impassable ground (nearest-vertex mask lookup). Steep ground
    /// only — the river is wadeable and never in the mask. The bridge
    /// deck rectangle is exempt so the deck stays walkable even if a
    /// mask rule ever covers the channel again.
    #[inline]
    pub fn blocked_at(&self, x: f32, z: f32) -> bool {
        if bridge_deck_contains(x, z) {
            return false;
        }
        let g = (Vec2::new(x, z) - self.origin) / CELL;
        let gx = (g.x.round() as usize).min(VERTS_X - 1);
        let gz = (g.y.round() as usize).min(VERTS_Z - 1);
        self.blocked[gz * VERTS_X + gx]
    }

    /// Wading slow multiplier: WADE_SLOW inside the channel, 1.0 on
    /// land — and on the bridge deck, which stays dry. Applied next to
    /// the slope penalty in the movement pass.
    #[inline]
    pub fn wade_mult(&self, x: f32, z: f32) -> f32 {
        if self.classic {
            return 1.0;
        }
        let d = (x - river_center_x(z)).abs();
        if d < river_half_width(z) && !bridge_deck_contains(x, z) {
            WADE_SLOW
        } else {
            1.0
        }
    }

    /// Carve a crater: smooth depression + small raised rim. Marks chunks dirty.
    pub fn carve_crater(&mut self, center: Vec2, radius: f32, depth: f32) {
        let rim_r = radius * 1.35;
        let gmin = ((center - rim_r - self.origin) / CELL).floor();
        let gmax = ((center + rim_r - self.origin) / CELL).ceil();
        let x0 = (gmin.x.max(0.0)) as usize;
        let z0 = (gmin.y.max(0.0)) as usize;
        let x1 = (gmax.x as usize).min(VERTS_X - 1);
        let z1 = (gmax.y as usize).min(VERTS_Z - 1);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let p = self.origin + Vec2::new(x as f32, z as f32) * CELL;
                let d = p.distance(center);
                let dh = if d < radius {
                    let t = 1.0 - (d / radius) * (d / radius);
                    -depth * t * t
                } else if d < rim_r {
                    let t = (d - radius) / (rim_r - radius);
                    depth * 0.12 * (1.0 - t * t)
                } else {
                    continue;
                };
                self.heights[z * VERTS_X + x] += dh;
            }
        }
        // Refresh the blocked mask over the touched area (one vertex of
        // margin: the slope rule reads 4-neighbors). The classic map
        // never blocks.
        if !self.classic {
            let bx0 = x0.saturating_sub(1);
            let bz0 = z0.saturating_sub(1);
            let bx1 = (x1 + 1).min(VERTS_X - 1);
            let bz1 = (z1 + 1).min(VERTS_Z - 1);
            for z in bz0..=bz1 {
                for x in bx0..=bx1 {
                    self.blocked[z * VERTS_X + x] = vertex_blocked(&self.heights, x, z);
                }
            }
        }
        // Normals read one neighbor beyond each vertex. Include both copies
        // of boundary vertices whose stencil touches the deformed region.
        let cx0 = x0.saturating_sub(2) / CHUNK_CELLS;
        let cz0 = z0.saturating_sub(2) / CHUNK_CELLS;
        let cx1 = ((x1 + 1) / CHUNK_CELLS).min(CHUNKS_X - 1);
        let cz1 = ((z1 + 1) / CHUNK_CELLS).min(CHUNKS_Z - 1);
        for cz in cz0..=cz1 {
            for cx in cx0..=cx1 {
                self.dirty[cz * CHUNKS_X + cx] = true;
            }
        }
    }

    /// March a ray to the surface. Returns the hit point.
    pub fn raycast(&self, ray: Ray3d) -> Option<Vec3> {
        let mut t = 0.0f32;
        let mut prev_t = 0.0f32;
        let dir = ray.direction.as_vec3();
        for _ in 0..1500 {
            let p = ray.origin + dir * t;
            if p.y < self.height_at(p.x, p.z) {
                // Bisect between prev_t and t.
                let (mut lo, mut hi) = (prev_t, t);
                for _ in 0..10 {
                    let mid = 0.5 * (lo + hi);
                    let q = ray.origin + dir * mid;
                    if q.y < self.height_at(q.x, q.z) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(ray.origin + dir * (0.5 * (lo + hi)));
            }
            prev_t = t;
            t += 1.5;
            if t > 2500.0 {
                break;
            }
        }
        None
    }
}

/// Mask rule per vertex: a face steeper than SLOPE_BLOCK against any
/// 4-neighbor (terrace risers, gorge walls, crater lips). The river
/// itself is wadeable and never blocks.
fn vertex_blocked(heights: &[f32], x: usize, z: usize) -> bool {
    let h = heights[z * VERTS_X + x];
    let mut max_d = 0.0f32;
    if x > 0 {
        max_d = max_d.max((h - heights[z * VERTS_X + x - 1]).abs());
    }
    if x + 1 < VERTS_X {
        max_d = max_d.max((h - heights[z * VERTS_X + x + 1]).abs());
    }
    if z > 0 {
        max_d = max_d.max((h - heights[(z - 1) * VERTS_X + x]).abs());
    }
    if z + 1 < VERTS_Z {
        max_d = max_d.max((h - heights[(z + 1) * VERTS_X + x]).abs());
    }
    max_d / CELL >= SLOPE_BLOCK
}

/// Original heights identify exposed crater soil without changing the sim surface.
#[derive(Resource, Default)]
struct TerrainChunks {
    meshes: Vec<Handle<Mesh>>,
    original_heights: Vec<f32>,
    material: Option<Handle<GroundMaterial>>,
    band_material: Option<Handle<StandardMaterial>>,
    /// Every map's textures, held so a map switch finds them loaded.
    warm: Vec<Handle<Image>>,
}

const GROUND_TEXTURES: [&str; 8] = [
    "terrain/pasture_natural_color.ktx2",
    "terrain/pasture_color.ktx2",
    "terrain/pasture_normal_roughness.ktx2",
    "terrain/earth_color.ktx2",
    "terrain/earth_normal_roughness.ktx2",
    "terrain/stony_soil_color.ktx2",
    "terrain/stony_soil_normal_roughness.ktx2",
    "terrain/grassland_layout_color.ktx2",
];

type GroundMaterial = ExtendedMaterial<StandardMaterial, GroundLayers>;

/// One terrain chunk entity.
#[derive(Component)]
struct GroundChunk;

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
struct GroundLayers {
    #[texture(100)]
    #[sampler(101)]
    pasture: Handle<Image>,
    #[texture(102)]
    pasture_normal: Handle<Image>,
    #[texture(103)]
    stone: Handle<Image>,
    #[texture(104)]
    stone_normal: Handle<Image>,
    #[texture(105)]
    earth: Handle<Image>,
    #[texture(106)]
    earth_normal: Handle<Image>,
    #[texture(107)]
    #[sampler(108)]
    coverage: Handle<Image>,
    #[uniform(109)]
    coverage_bounds: Vec4,
    #[uniform(110)]
    natural_ground: u32,
}

impl MaterialExtension for GroundLayers {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain.wgsl".into()
    }
}

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainChunks>()
            .add_message::<MapChanged>()
            .add_plugins(MaterialPlugin::<GroundMaterial>::default())
            .add_systems(PreStartup, generate_terrain)
            .add_systems(Startup, spawn_chunks)
            .add_systems(
                Update,
                (rebuild_map.in_set(MapRebuild), crater_tool, auto_test_craters, remesh_dirty).chain(),
            );
    }
}

pub(crate) fn fbm(p: Vec2) -> f32 {
    fn lattice(xi: i32, zi: i32) -> f32 {
        let ux = xi as u32;
        let uz = zi as u32;
        hash01(ux.wrapping_mul(0x9E37_79B1) ^ uz.wrapping_mul(0x85EB_CA77))
    }
    fn value_noise(p: Vec2) -> f32 {
        let x0 = p.x.floor();
        let z0 = p.y.floor();
        let fx = p.x - x0;
        let fz = p.y - z0;
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sz = fz * fz * (3.0 - 2.0 * fz);
        let (xi, zi) = (x0 as i32, z0 as i32);
        let v00 = lattice(xi, zi);
        let v10 = lattice(xi + 1, zi);
        let v01 = lattice(xi, zi + 1);
        let v11 = lattice(xi + 1, zi + 1);
        let a = v00 + (v10 - v00) * sx;
        let b = v01 + (v11 - v01) * sx;
        (a + (b - a) * sz) * 2.0 - 1.0
    }
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for _ in 0..4 {
        sum += value_noise(p * freq) * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

// ---- River -------------------------------------------------------------
// A shallow, fordable river meandering north-south through the midfield.
// Purely visual for now: the bed is carved into the heightfield and the
// water surface (water.rs) follows the same three functions.

/// River centerline: x as a function of z.
pub fn river_center_x(z: f32) -> f32 {
    110.0 * (z * 0.0055 + 0.9).sin() + 55.0 * (z * 0.0017 + 2.6).sin()
}

/// Channel half-width (to the shoreline), varies along the course.
pub fn river_half_width(z: f32) -> f32 {
    14.0 + 4.0 * (z * 0.011 + 0.5).sin()
}

/// Water surface height, gently falling from north to south.
pub fn river_water_level(z: f32) -> f32 {
    0.5 + z * 0.008
}

/// Max bed depth below the water surface (channel center): wading
/// depth — the whole river is fordable, just slow (M2TW shallows).
pub const RIVER_DEPTH: f32 = 0.6;
/// Bank rise per unit t (t = distance / half-width) past the shoreline.
pub(crate) const RIVER_BANK: f32 = 3.0;
/// River-shaped corridor half-width in units of t (beyond: untouched).
pub const RIVER_CORRIDOR: f32 = 2.3;

// ---- Bridge -------------------------------------------------------------
// One stone bridge (south half): a solid deck over the channel. Not a
// required crossing — the river is wadeable everywhere — but units on
// the deck stay dry and keep full speed, so it emerges as the fast
// route with zero pathfinding.

/// z of the bridge crossing (south half).
pub const BRIDGE_Z: f32 = 130.0;
/// Half-width of the bridge deck along z (the walkable band).
pub const BRIDGE_HALF_SPAN: f32 = 6.0;
/// Deck height above the local water level.
pub const BRIDGE_DECK_LIFT: f32 = 1.5;

/// Slope (rise per meter) at which ground becomes impassable: terrace
/// risers, gorge walls, crater lips. Below this the slope penalty only.
pub const SLOPE_BLOCK: f32 = 1.0;
/// Speed multiplier while wading the channel.
pub const WADE_SLOW: f32 = 0.55;

/// Deck half-length across the channel: it lands where the carved
/// bank profile reaches deck height (wl + (t−1)·BANK = wl + LIFT).
pub fn bridge_deck_half_len(z: f32) -> f32 {
    (1.0 + BRIDGE_DECK_LIFT / RIVER_BANK) * river_half_width(z)
}

/// The bridge deck rectangle (x across the channel, z along it).
/// Units inside it walk at deck height and are never "in the water".
pub fn bridge_deck_contains(x: f32, z: f32) -> bool {
    (z - BRIDGE_Z).abs() < BRIDGE_HALF_SPAN
        && (x - river_center_x(z)).abs() < bridge_deck_half_len(z)
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Riverbed depth below the water at normalized distance t (0 at shore).
pub fn river_bed_depth(t: f32) -> f32 {
    RIVER_DEPTH * (1.0 - t * t).max(0.0).powf(0.75)
}

/// Broad asymmetric shoulders frame an open lowland. Analytic falloffs avoid
/// ridge cusps and circular slope breaks through the deployment area.
fn classic_height(p: Vec2) -> f32 {
    fn shoulder(p: Vec2, center: Vec2, radii: Vec2, angle: f32, height: f32) -> f32 {
        let q = Mat2::from_angle(angle) * (p - center) / radii;
        height * (-q.length_squared()).exp()
    }
    let west = shoulder(
        p,
        Vec2::new(-390.0, -100.0),
        Vec2::new(190.0, 430.0),
        0.2,
        17.0,
    );
    let east = shoulder(
        p,
        Vec2::new(380.0, 160.0),
        Vec2::new(230.0, 410.0),
        -0.45,
        13.0,
    );
    let north = shoulder(
        p,
        Vec2::new(-60.0, -390.0),
        Vec2::new(450.0, 170.0),
        0.15,
        8.0,
    );
    let south = shoulder(
        p,
        Vec2::new(80.0, 430.0),
        Vec2::new(420.0, 240.0),
        -0.2,
        4.0,
    );
    let valley = (p.x - 0.18 * p.y - 30.0 * (p.y / 240.0).sin()) / 155.0;
    let detail = fbm(Mat2::from_angle(0.61) * p / 190.0 + Vec2::new(11.8, 46.2)) * 0.65;
    5.0 + p.x * 0.002 - p.y * 0.003 + west + east + north + south - 3.0 * (-valley * valley).exp()
        + detail
}

fn generate_terrain(mut commands: Commands) {
    commands.insert_resource(build_terrain(MapKind::from_env()));
}

/// The heightfield and the blocked mask of one map. Grassland has its
/// own analytic landforms; classic and river share the noise formula,
/// and river alone carves the channel, terraces and impassable walls.
fn build_terrain(kind: MapKind) -> Terrain {
    let classic = kind != MapKind::River;
    let origin = Vec2::new(
        -(VERTS_X as f32 - 1.0) * CELL * 0.5,
        -(VERTS_Z as f32 - 1.0) * CELL * 0.5,
    );
    let mut heights = vec![0.0f32; VERTS_X * VERTS_Z];
    for z in 0..VERTS_Z {
        for x in 0..VERTS_X {
            let p = origin + Vec2::new(x as f32, z as f32) * CELL;
            if kind == MapKind::Grassland {
                heights[z * VERTS_X + x] = classic_height(p);
                continue;
            }
            // Experimental river base: rolling landforms and ridged peaks.
            let base = fbm(p / 320.0) * 22.0;
            let detail = fbm(p / 90.0 + Vec2::splat(37.7)) * 4.5;
            let r = 1.0 - fbm(p / 260.0 + Vec2::splat(91.3)).abs().min(1.0);
            let ridged = r * r * 16.0;
            let mut h = base + detail + ridged;
            // Soften (not flatten) the central battlefield: rolling and
            // readable in the middle, dramatic on the outskirts.
            let center_dist = (p.length() - 140.0).max(0.0) / 260.0;
            h *= 0.35 + 0.65 * center_dist.min(1.0);
            // Bias up so the battlefield sits in the grass bands; dirt only
            // in real hollows and crater floors.
            h += 2.6;
            if !classic {
                // Terraced highlands: partially quantize the TALL ground
                // into 3.5 m steps — flat shading turns the risers into
                // angled plateau faces, and the blocked mask makes those
                // massifs impassable (M2TW mountains framing the field).
                // Mid-height hills stay smooth and traversable.
                let ts = smoothstep(15.0, 26.0, h);
                if ts > 0.0 {
                    const STEP: f32 = 3.5;
                    let hq = (h / STEP).round() * STEP;
                    h += (hq - h) * ts * 0.8;
                }
                // River channel: exact bed profile inside the shoreline,
                // banks blended out to t = RIVER_CORRIDOR. Carving after
                // everything else so the river always wins; at the mountainous
                // map edges this cuts a gorge.
                let d = (p.x - river_center_x(p.y)).abs();
                let t = d / river_half_width(p.y);
                if t < RIVER_CORRIDOR {
                    let wl = river_water_level(p.y);
                    let prof = if t <= 1.0 {
                        wl - river_bed_depth(t)
                    } else {
                        wl + (t - 1.0) * RIVER_BANK
                    };
                    let s = smoothstep(1.0, RIVER_CORRIDOR, t);
                    h = prof + (h - prof) * s;
                }
            }
            heights[z * VERTS_X + x] = h;
        }
    }
    let mut blocked = vec![false; VERTS_X * VERTS_Z];
    if !classic {
        for z in 0..VERTS_Z {
            for x in 0..VERTS_X {
                blocked[z * VERTS_X + x] = vertex_blocked(&heights, x, z);
            }
        }
    }
    info!("terrain: {} map", kind.label());
    Terrain {
        heights,
        blocked,
        kind,
        classic,
        origin,
        dirty: vec![false; CHUNKS_X * CHUNKS_Z],
    }
}

fn ground_texture(assets: &AssetServer, path: &'static str) -> Handle<Image> {
    assets
        .load_builder()
        .with_settings(move |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = path.ends_with("_color.ktx2");
            let address_mode = if path == "terrain/grassland_layout_color.ktx2" {
                ImageAddressMode::ClampToEdge
            } else {
                ImageAddressMode::Repeat
            };
            settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                anisotropy_clamp: 8,
                ..ImageSamplerDescriptor::linear()
            });
        })
        .load(path)
}

fn ground_base() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.95,
        reflectance: 0.05,
        ..default()
    }
}

/// The ground material's layers for one map: the grassland gets the
/// authored layout and the natural pasture, the classic field and the
/// river map get procedural coverage from their relief and stony slopes.
fn ground_layers(assets: &AssetServer, images: &mut Assets<Image>, terrain: &Terrain) -> GroundLayers {
    // Only the grassland has a painted layout; the other maps derive
    // their coverage from their own relief and put stone on steep faces.
    let authored = terrain.kind == MapKind::Grassland;
    GroundLayers {
        pasture: ground_texture(
            assets,
            if authored {
                "terrain/pasture_natural_color.ktx2"
            } else {
                "terrain/pasture_color.ktx2"
            },
        ),
        pasture_normal: ground_texture(assets, "terrain/pasture_normal_roughness.ktx2"),
        stone: ground_texture(
            assets,
            if authored {
                "terrain/earth_color.ktx2"
            } else {
                "terrain/stony_soil_color.ktx2"
            },
        ),
        stone_normal: ground_texture(
            assets,
            if authored {
                "terrain/earth_normal_roughness.ktx2"
            } else {
                "terrain/stony_soil_normal_roughness.ktx2"
            },
        ),
        earth: ground_texture(assets, "terrain/earth_color.ktx2"),
        earth_normal: ground_texture(assets, "terrain/earth_normal_roughness.ktx2"),
        coverage: if authored {
            ground_texture(assets, "terrain/grassland_layout_color.ktx2")
        } else {
            images.add(ground_coverage(terrain))
        },
        coverage_bounds: if authored {
            Vec4::new(
                terrain.origin.x,
                terrain.origin.y,
                (VERTS_X - 1) as f32 * CELL,
                (VERTS_Z - 1) as f32 * CELL,
            )
        } else {
            Vec4::new(
                terrain.origin.x - CELL * 0.5,
                terrain.origin.y - CELL * 0.5,
                VERTS_X as f32 * CELL,
                VERTS_Z as f32 * CELL,
            )
        },
        natural_ground: u32::from(authored),
    }
}

/// The menu changed the map: regenerate the heightfield in place, give
/// every chunk a material with the new map's layers and mark every
/// chunk, so `remesh_dirty` rebuilds the whole field later in this frame.
#[allow(clippy::too_many_arguments)]
fn rebuild_map(
    mut changes: MessageReader<MapChanged>,
    mut terrain: ResMut<Terrain>,
    mut chunks: ResMut<TerrainChunks>,
    mut materials: ResMut<Assets<GroundMaterial>>,
    assets: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    chunk_entities: Query<Entity, With<GroundChunk>>,
) {
    let Some(MapChanged(kind)) = changes.read().last().copied() else {
        return;
    };
    if kind == terrain.kind {
        return;
    }
    let t0 = Instant::now();
    *terrain = build_terrain(kind);
    terrain.dirty.fill(true);
    chunks.original_heights.clone_from(&terrain.heights);
    // A fresh material on every chunk: a component change, which the
    // renderer tracks, where editing the asset in place did not reach them.
    let material = materials.add(GroundMaterial {
        base: ground_base(),
        extension: ground_layers(&assets, &mut images, &terrain),
    });
    for e in &chunk_entities {
        let mut entity = commands.entity(e);
        if kind == MapKind::Classic {
            entity.remove::<MeshMaterial3d<GroundMaterial>>();
            if let Some(band) = &chunks.band_material {
                entity.insert(MeshMaterial3d(band.clone()));
            }
        } else {
            entity.remove::<MeshMaterial3d<StandardMaterial>>();
            entity.insert(MeshMaterial3d(material.clone()));
        }
    }
    chunks.material = Some(material);
    debug!(
        "map rebuilt as {} in {:.2} ms",
        kind.label(),
        t0.elapsed().as_secs_f32() * 1000.0
    );
}

#[allow(clippy::too_many_arguments)]
fn spawn_chunks(
    mut commands: Commands,
    terrain: Res<Terrain>,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GroundMaterial>>,
    mut standard: ResMut<Assets<StandardMaterial>>,
    mut chunks: ResMut<TerrainChunks>,
    mut images: ResMut<Assets<Image>>,
) {
    let material = materials.add(GroundMaterial {
        base: ground_base(),
        extension: ground_layers(&assets, &mut images, &terrain),
    });
    chunks.material = Some(material.clone());
    let band_material = standard.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        reflectance: 0.05,
        ..default()
    });
    chunks.band_material = Some(band_material.clone());
    // Otherwise a switch in the menu drops the ground for the frames the
    // new map's textures take to load.
    chunks.warm = GROUND_TEXTURES
        .iter()
        .copied()
        .map(|path| ground_texture(&assets, path))
        .collect();
    chunks.original_heights.clone_from(&terrain.heights);
    for cz in 0..CHUNKS_Z {
        for cx in 0..CHUNKS_X {
            let mesh = build_chunk_mesh(&terrain, &chunks.original_heights, cx, cz);
            let aabb = mesh.compute_aabb();
            let handle = meshes.add(mesh);
            chunks.meshes.push(handle.clone());
            let mut e = commands.spawn((Mesh3d(handle), GroundChunk));
            if terrain.kind == MapKind::Classic {
                e.insert(MeshMaterial3d(band_material.clone()));
            } else {
                e.insert(MeshMaterial3d(material.clone()));
            }
            if let Some(aabb) = aabb {
                e.insert(aabb);
            }
        }
    }
}

/// Central differences use the global grid, including across chunk boundaries.
fn ground_normal(terrain: &Terrain, x: usize, z: usize) -> Vec3 {
    let xm = x.saturating_sub(1);
    let xp = (x + 1).min(VERTS_X - 1);
    let zm = z.saturating_sub(1);
    let zp = (z + 1).min(VERTS_Z - 1);
    let dx = (terrain.h(xp, z) - terrain.h(xm, z)) / ((xp - xm) as f32 * CELL);
    let dz = (terrain.h(x, zp) - terrain.h(x, zm)) / ((zp - zm) as f32 * CELL);
    Vec3::new(-dx, 1.0, -dz).normalize()
}

/// Smooth gradient noise for rendering only; does not draw from simulation RNGs.
fn cover_noise(p: Vec2) -> f32 {
    fn gradient(x: i32, y: i32) -> Vec2 {
        let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841);
        h ^= h >> 16;
        h = h.wrapping_mul(0x7feb_352d);
        h ^= h >> 15;
        let a = h as f32 * (std::f32::consts::TAU / u32::MAX as f32);
        Vec2::new(a.cos(), a.sin())
    }
    let base = p.floor();
    let f = p - base;
    let u = f * f * f * (f * (f * 6.0 - Vec2::splat(15.0)) + Vec2::splat(10.0));
    let x = base.x as i32;
    let y = base.y as i32;
    let a = gradient(x, y).dot(f);
    let b = gradient(x + 1, y).dot(f - Vec2::X);
    let c = gradient(x, y + 1).dot(f - Vec2::Y);
    let d = gradient(x + 1, y + 1).dot(f - Vec2::ONE);
    ((a + (b - a) * u.x) * (1.0 - u.y) + (c + (d - c) * u.x) * u.y) * 1.5
}

/// One clamped field spans the battlefield. Warped coordinates break lattice
/// alignment; relative relief keeps hollows greener than nearby shoulders.
fn ground_coverage(terrain: &Terrain) -> Image {
    let mut pixels = Vec::with_capacity(VERTS_X * VERTS_Z * 4);
    for z in 0..VERTS_Z {
        for x in 0..VERTS_X {
            let p = terrain.origin + Vec2::new(x as f32, z as f32) * CELL;
            let warp = Vec2::new(
                cover_noise(p / 170.0 + Vec2::new(17.2, 81.7)),
                cover_noise(p / 170.0 + Vec2::new(53.6, 11.3)),
            ) * 65.0;
            let q = Mat2::from_angle(0.57) * (p + warp);
            let broad = cover_noise(q / Vec2::new(145.0, 85.0) + Vec2::splat(7.3));
            let patches = cover_noise(q / 31.0 + Vec2::splat(37.8));
            let flecks = cover_noise(q / 7.0 + Vec2::splat(91.1));
            let neighbors = terrain.h(x.saturating_sub(48), z)
                + terrain.h((x + 48).min(VERTS_X - 1), z)
                + terrain.h(x, z.saturating_sub(48))
                + terrain.h(x, (z + 48).min(VERTS_Z - 1));
            let relief = terrain.h(x, z) - neighbors * 0.25;
            let dry = (0.52 + broad * 0.28 + patches * 0.08 + relief * 0.07).clamp(0.0, 1.0);
            let soil = smoothstep(0.52, 0.82, dry + patches * 0.16) * 0.5;
            let variation = (0.5 + flecks * 0.08 + patches * 0.06).clamp(0.0, 1.0);
            for value in [dry, soil, variation, 1.0] {
                pixels.push((value * 255.0).round() as u8);
            }
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: VERTS_X as u32,
            height: VERTS_Z as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

/// RGBA carries an unused channel, crater exposure, stone and bank dampness.
fn ground_weights(terrain: &Terrain, original: &[f32], x: usize, z: usize, n: Vec3) -> [f32; 4] {
    let p = terrain.origin + Vec2::new(x as f32, z as f32) * CELL;
    let h = terrain.h(x, z);
    let slope = n.xz().length() / n.y.max(0.001);
    let disturbed = smoothstep(0.03, 0.65, (h - original[z * VERTS_X + x]).abs());
    let mut soil = disturbed;
    let stone = smoothstep(0.45, 0.95, slope) * (1.0 - disturbed * 0.8);
    let mut damp = 0.0;
    if !terrain.classic {
        let bank_distance = (p.x - river_center_x(p.y)).abs() - river_half_width(p.y);
        damp = 1.0 - smoothstep(0.0, 10.0, bank_distance);
        soil = soil.max(damp * 0.8);
    }
    [0.0, soil, stone, damp]
}

fn band_color(h: f32, slope: f32, p: Vec2, classic: bool) -> [f32; 4] {
    let mut c = if slope > 0.75 {
        Color::srgb(0.46, 0.42, 0.36) // scree on steep faces
    } else if h < -2.5 {
        Color::srgb(0.33, 0.25, 0.17) // crater floor / deep dirt
    } else if h < 0.0 {
        Color::srgb(0.43, 0.34, 0.22) // dirt
    } else if h < 5.0 {
        Color::srgb(0.34, 0.43, 0.22) // low grass
    } else if h < 11.0 {
        Color::srgb(0.42, 0.50, 0.26) // grass
    } else if h < 17.0 {
        Color::srgb(0.52, 0.52, 0.33) // dry highland
    } else if h < 24.0 {
        Color::srgb(0.52, 0.48, 0.42) // rock
    } else {
        Color::srgb(0.78, 0.79, 0.82) // snowcap
    };
    // Grass-band variety: golden wheat patches and a subtle per-triangle
    // tone wobble so the open field doesn't read as flat plastic.
    // (New map only — the classic map keeps the flat bands.)
    if !classic && slope <= 0.75 && (0.0..11.0).contains(&h) {
        let patch = fbm(p / 70.0 + Vec2::splat(47.1));
        if patch > 0.62 {
            c = Color::srgb(0.62, 0.53, 0.24); // wheat field
        } else {
            let k = 0.94 + 0.12 * fbm(p / 45.0 + Vec2::splat(13.7));
            let l = c.to_linear();
            c = Color::linear_rgb(l.red * k, l.green * k, l.blue * k);
        }
    }
    c.to_linear().to_f32_array()
}
/// Flat-shaded triangle soup for one chunk: 2 triangles per cell, per-face
/// normal and one hard-banded color per triangle. World coords baked in.
fn build_band_mesh(terrain: &Terrain, cx: usize, cz: usize) -> Mesh {
    let n_tris = CHUNK_CELLS * CHUNK_CELLS * 2;
    let mut positions = Vec::with_capacity(n_tris * 3);
    let mut normals = Vec::with_capacity(n_tris * 3);
    let mut colors = Vec::with_capacity(n_tris * 3);

    let vx0 = cx * CHUNK_CELLS;
    let vz0 = cz * CHUNK_CELLS;
    for dz in 0..CHUNK_CELLS {
        for dx in 0..CHUNK_CELLS {
            let (x, z) = (vx0 + dx, vz0 + dz);
            let wp = |xx: usize, zz: usize| -> Vec3 {
                let w = terrain.origin + Vec2::new(xx as f32, zz as f32) * CELL;
                Vec3::new(w.x, terrain.h(xx, zz), w.y)
            };
            let p00 = wp(x, z);
            let p10 = wp(x + 1, z);
            let p01 = wp(x, z + 1);
            let p11 = wp(x + 1, z + 1);
            // Alternate the quad split diagonal for a less regular look.
            let tris = if (x + z) % 2 == 0 {
                [[p00, p01, p11], [p00, p11, p10]]
            } else {
                [[p00, p01, p10], [p10, p01, p11]]
            };
            for tri in tris {
                let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
                let hc = (tri[0].y + tri[1].y + tri[2].y) / 3.0;
                let slope = (1.0 - n.y * n.y).sqrt() / n.y.max(0.1);
                let pc = Vec2::new(
                    (tri[0].x + tri[1].x + tri[2].x) / 3.0,
                    (tri[0].z + tri[1].z + tri[2].z) / 3.0,
                );
                let col = band_color(hc, slope, pc, true);
                for v in tri {
                    positions.push(v);
                    normals.push(n);
                    colors.push(col);
                }
            }
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
}

/// The classic map keeps its 0.1.0 look: flat shading and height bands.
fn build_chunk_mesh(terrain: &Terrain, original: &[f32], cx: usize, cz: usize) -> Mesh {
    if terrain.kind == MapKind::Classic {
        return build_band_mesh(terrain, cx, cz);
    }
    let side = CHUNK_CELLS + 1;
    let mut positions = Vec::with_capacity(side * side);
    let mut normals = Vec::with_capacity(side * side);
    let mut weights = Vec::with_capacity(side * side);
    let mut indices = Vec::with_capacity(CHUNK_CELLS * CHUNK_CELLS * 6);
    let vx0 = cx * CHUNK_CELLS;
    let vz0 = cz * CHUNK_CELLS;
    for dz in 0..side {
        for dx in 0..side {
            let (x, z) = (vx0 + dx, vz0 + dz);
            let p = terrain.origin + Vec2::new(x as f32, z as f32) * CELL;
            let n = ground_normal(terrain, x, z);
            positions.push([p.x, terrain.h(x, z), p.y]);
            normals.push(n.to_array());
            weights.push(ground_weights(terrain, original, x, z, n));
        }
    }
    for dz in 0..CHUNK_CELLS {
        for dx in 0..CHUNK_CELLS {
            let a = (dz * side + dx) as u32;
            let b = a + 1;
            let c = a + side as u32;
            let d = c + 1;
            // Preserve the heightfield's alternating triangle split.
            if (vx0 + dx + vz0 + dz).is_multiple_of(2) {
                indices.extend_from_slice(&[a, c, d, a, d, b]);
            } else {
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, weights)
    .with_inserted_indices(Indices::U32(indices))
}

fn remesh_dirty(
    mut terrain: ResMut<Terrain>,
    chunks: Res<TerrainChunks>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_entities: Query<(&Mesh3d, &mut Aabb)>,
) {
    if !terrain.dirty.iter().any(|d| *d) {
        return;
    }
    let t0 = Instant::now();
    let mut rebuilt = 0;
    for cz in 0..CHUNKS_Z {
        for cx in 0..CHUNKS_X {
            let ci = cz * CHUNKS_X + cx;
            if !terrain.dirty[ci] {
                continue;
            }
            let mesh = build_chunk_mesh(&terrain, &chunks.original_heights, cx, cz);
            let aabb = mesh.compute_aabb();
            let _ = meshes.insert(&chunks.meshes[ci], mesh);
            if let Some(new_aabb) = aabb {
                for (m, mut old) in &mut chunk_entities {
                    if m.0 == chunks.meshes[ci] {
                        *old = new_aabb;
                    }
                }
            }
            rebuilt += 1;
        }
    }
    terrain.dirty.fill(false);
    debug!(
        "remeshed {rebuilt} chunks in {:.2} ms",
        t0.elapsed().as_secs_f32() * 1000.0
    );
}

/// Debug tool: X carves a crater under the cursor.
fn crater_tool(
    keys: Res<ButtonInput<KeyCode>>,
    mut terrain: ResMut<Terrain>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    time: Res<Time>,
    mut cooldown: Local<f32>,
) {
    *cooldown -= time.delta_secs();
    if !keys.pressed(KeyCode::KeyX) || *cooldown > 0.0 {
        return;
    }
    let Ok(window) = window.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        return;
    };
    if let Some(hit) = terrain.raycast(ray) {
        terrain.carve_crater(Vec2::new(hit.x, hit.z), 11.0, 5.0);
        *cooldown = 0.15;
        info!("crater at ({:.0}, {:.0})", hit.x, hit.z);
    }
}

/// FL_TEST_CRATERS=1: carve a crater near the field center every 2 s
/// (screenshot/perf verification without input injection).
fn auto_test_craters(
    mut terrain: ResMut<Terrain>,
    time: Res<Time>,
    mut next: Local<f32>,
    mut n: Local<u32>,
) {
    if std::env::var("FL_TEST_CRATERS").is_err() {
        return;
    }
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 2.0;
    *n += 1;
    let a = hash01(*n * 7 + 1) * std::f32::consts::TAU;
    let r = hash01(*n * 7 + 2).sqrt() * 140.0;
    let center = Vec2::new(a.cos(), a.sin()) * r;
    let radius = 8.0 + hash01(*n * 7 + 3) * 6.0;
    terrain.carve_crater(center, radius, radius * 0.45);
    info!(
        "test crater #{} at ({:.0}, {:.0}) r={radius:.1}",
        *n, center.x, center.y
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    #[test]
    fn classic_landforms_keep_gentle_slopes_without_sharp_cell_ridges() {
        let mut min_height = f32::INFINITY;
        let mut max_height = f32::NEG_INFINITY;
        let mut max_slope = 0.0_f32;
        let mut max_bend = 0.0_f32;
        for z in 1..VERTS_Z - 1 {
            for x in 1..VERTS_X - 1 {
                let p = Vec2::new(
                    x as f32 - (VERTS_X - 1) as f32 * 0.5,
                    z as f32 - (VERTS_Z - 1) as f32 * 0.5,
                ) * CELL;
                let h = classic_height(p);
                min_height = min_height.min(h);
                max_height = max_height.max(h);
                let dx = (classic_height(p + Vec2::X * CELL) - classic_height(p - Vec2::X * CELL))
                    / (2.0 * CELL);
                let dz = (classic_height(p + Vec2::Y * CELL) - classic_height(p - Vec2::Y * CELL))
                    / (2.0 * CELL);
                max_slope = max_slope.max(Vec2::new(dx, dz).length());
                for axis in [Vec2::X, Vec2::Y] {
                    let bend =
                        classic_height(p + axis * CELL) - 2.0 * h + classic_height(p - axis * CELL);
                    max_bend = max_bend.max(bend.abs());
                }
            }
        }
        println!(
            "classic heights {min_height:.2}..{max_height:.2} m, max slope {max_slope:.4}, max cell bend {max_bend:.4} m"
        );
        assert!(min_height.is_finite() && max_height.is_finite());
        assert!(max_height - min_height > 10.0);
        assert!(max_slope < 0.2, "plain slopes exceed a 20% grade");
        assert!(max_bend < 0.1, "adjacent cells form a sharp ridge");
    }

    fn sloped_field() -> Terrain {
        let mut heights = vec![0.0; VERTS_X * VERTS_Z];
        for z in 0..VERTS_Z {
            for x in 0..VERTS_X {
                heights[z * VERTS_X + x] = (x as f32 * 0.125 + z as f32 * 0.25) * CELL;
            }
        }
        Terrain {
            heights,
            blocked: vec![false; VERTS_X * VERTS_Z],
            kind: MapKind::Grassland,
            classic: true,
            origin: Vec2::ZERO,
            dirty: vec![false; CHUNKS_X * CHUNKS_Z],
        }
    }

    fn vec3_attribute(mesh: &Mesh, attribute: bevy::mesh::MeshVertexAttribute) -> &Vec<[f32; 3]> {
        let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(attribute) else {
            panic!("expected a three-component vertex attribute");
        };
        values
    }

    #[test]
    fn indexed_ground_preserves_heights_winding_and_chunk_seams() {
        let terrain = sloped_field();
        let left = build_chunk_mesh(&terrain, &terrain.heights, 0, 0);
        let right = build_chunk_mesh(&terrain, &terrain.heights, 1, 0);
        let positions = vec3_attribute(&left, Mesh::ATTRIBUTE_POSITION);
        let normals = vec3_attribute(&left, Mesh::ATTRIBUTE_NORMAL);
        let right_positions = vec3_attribute(&right, Mesh::ATTRIBUTE_POSITION);
        let right_normals = vec3_attribute(&right, Mesh::ATTRIBUTE_NORMAL);
        assert_eq!(positions.len(), 1089);
        assert_eq!(left.indices().unwrap().len(), 6144);
        let expected_normal = Vec3::new(-0.125, 1.0, -0.25).normalize();
        for (i, position) in positions.iter().enumerate() {
            let x = i % 33;
            let z = i / 33;
            assert_eq!(position[1], terrain.h(x, z));
            assert!(Vec3::from_array(normals[i]).distance(expected_normal) < 1e-6);
        }
        let indices: Vec<_> = left.indices().unwrap().iter().collect();
        for tri in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[tri[0]]);
            let b = Vec3::from_array(positions[tri[1]]);
            let c = Vec3::from_array(positions[tri[2]]);
            assert!((b - a).cross(c - a).y > 0.0);
        }
        for z in 0..33 {
            assert_eq!(positions[z * 33 + 32], right_positions[z * 33]);
            assert_eq!(normals[z * 33 + 32], right_normals[z * 33]);
        }
    }

    #[test]
    fn crater_exposes_soil_and_invalidates_every_changed_normal() {
        let mut terrain = sloped_field();
        let before = terrain.clone();
        terrain.carve_crater(Vec2::new(66.0, 66.0), 5.0, 2.0);
        for cz in 0..2 {
            for cx in 0..2 {
                for z in cz * CHUNK_CELLS..=(cz + 1) * CHUNK_CELLS {
                    for x in cx * CHUNK_CELLS..=(cx + 1) * CHUNK_CELLS {
                        if ground_normal(&terrain, x, z) != ground_normal(&before, x, z) {
                            assert!(terrain.dirty[cz * CHUNKS_X + cx]);
                        }
                    }
                }
            }
        }
        let weights = ground_weights(
            &terrain,
            &before.heights,
            33,
            33,
            ground_normal(&terrain, 33, 33),
        );
        assert_eq!(weights[1], 1.0);
        assert_eq!(
            terrain.height_at(66.0, 66.0),
            before.height_at(66.0, 66.0) - 2.0
        );
        assert!(terrain.blocked.iter().all(|blocked| !blocked));
    }
}
