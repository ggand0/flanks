#import bevy_pbr::mesh_view_bindings::{globals, lights, view}
#import bevy_pbr::mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT
#import bevy_pbr::shadows::fetch_directional_shadow
#import bevy_pbr::view_transformations::position_world_to_clip
#import flanks::unit_pose::{P_DEATH, P_POS, P_TEAM, P_YAW, place, pose_at}
#ifdef UNIT_POSE_READ
#import flanks::unit_pose::pose_from
#else
#import flanks::unit_pose::pose_begin
#endif

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // NOT texture coords: x = body part id (0 body, 1 sword arm,
    // 2 left leg, 3 right leg, 4 spear arm, 5 shield arm, 6 bow arm,
    // 7 arrow, 8 weapon, 9 and 10 a jointed weapon arm's forearm and
    // hand, and an archer's bow rig: 11 and 12 the bow forearm and hand,
    // 13 and 17 the string halves, 14 and 15 the limbs, 16 the held
    // arrow, 18 head, 19 torso, 20 the archer's knife), y = the part's
    // pivot height.
    @location(2) part_pivot: vec2<f32>,
    // Where the vertex samples the kind's atlas (zero without one).
    @location(3) atlas_uv: vec2<f32>,
    // Part material: rgb = fixed color, a = team-color blend amount.
    @location(5) v_color: vec4<f32>,

    // Per-instance: xyz = world position. w = an arrow's uniform scale,
    // or how far a soldier's bow is up from the carry toward the drawn
    // ready pose, 0 to 1 (render_units.rs InstanceData).
    @location(8) i_pos_scale: vec4<f32>,
    // rgb = team color, a = stable per-unit anim seed (not opacity).
    @location(9) i_color: vec4<f32>,
    // x = yaw, y = ground speed in m/s. z = attack or cheer
    // (render_units.rs CELEBRATE_BASE), 0 for neither. w = fx: [0,1]
    // hit-flash intensity, (1,2] = 1 + death progress, 2 on a corpse.
    @location(10) i_anim: vec4<f32>,
    // x = stance band (0.25 enemy near, 0.5 fighting, 1 charging),
    // y = wall 0..1 (shieldwall/spearwall by bucket), z = gait phase in
    // cycles, w = stagger.
    @location(11) i_anim2: vec4<f32>,
};

// Kept narrow on purpose: a far view pushes 60M vertices through this
// shader per frame, and every float here is paid that many times, while
// the fragment runs about eight million times (devlog 0147).
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // rgb = the lit vertex colour: part material blended with the team
    // colour, times the sun and the sky. a = the sun's share of that
    // light, the part the shadow takes away. Under UNIT_LIGHT_FLAT each
    // triangle takes it from its first vertex, which carries the
    // triangle's three corner normals (render_units_gpu.rs `weld`).
#ifdef UNIT_LIGHT_FLAT
    @location(0) @interpolate(flat, first) color: vec4<f32>,
#else
#ifdef UNIT_CENTROID
    // FL_CENTROID=1 (render_units.rs `centroid_defs`): under multisampling
    // a fragment of a triangle smaller than a pixel reads the blended
    // light at a sample the triangle covers, not at the pixel centre
    // outside it, where the blend runs past its corners' values.
    @location(0) @interpolate(perspective, centroid) color: vec4<f32>,
#else
    @location(0) color: vec4<f32>,
#endif
#endif
    // Per instance, so flat. x = team colour, unorm8 rgb, and the death
    // darkening in the top byte; y = the hit flash, unorm8 in the low byte,
    // and under FL_OCCLUDE_CHECK the occlusion mark in bit 8.
    @location(1) @interpolate(flat) packed: vec2<u32>,
#ifdef UNIT_SHADOW_RECEIVE
    // World position pushed off the surface by the shadow map's normal
    // bias: where the fragment samples the sun's shadow. Only buckets
    // whose soldiers can stand in a cascade carry it
    // (render_units_shadow.rs `ShadowReceiveLevels`).
    @location(2) shadow_position: vec3<f32>,
#endif
#ifdef UNIT_ATLAS
#ifdef UNIT_ATLAS_FLAT
    // One atlas point per triangle, from its first vertex: a welded far
    // level's vertex is shared by triangles that sample different points
    // (render_units_gpu.rs `weld`).
    @location(3) @interpolate(flat, first) atlas_uv: vec2<f32>,
#else
    @location(3) atlas_uv: vec2<f32>,
#endif
#endif
};

#ifdef UNIT_ATLAS
// The kind's atlas: rgb colour, alpha the team tint mask. Only buckets
// with an atlas compile this path (render_units.rs `atlas_defs`).
@group(3) @binding(4) var unit_atlas: texture_2d<f32>;
@group(3) @binding(5) var unit_atlas_sampler: sampler;
#endif

// The scene's sun as the vertex stage sees it (render_units_shadow.rs
// `SunUniform`): bevy binds its lights to the fragment stage only.
struct Sun {
    // xyz = direction toward the sun, zero without a light.
    direction_to_light: vec4<f32>,
    // x = 1 with shadow maps on, y = the light's shadow normal bias,
    // z = the cascade overlap proportion, w = the cascade count.
    params: vec4<f32>,
    // Per cascade: its far bound in view depth, and its texel in metres.
    far_bounds: vec4<f32>,
    texel_sizes: vec4<f32>,
};
@group(3) @binding(8) var<uniform> sun: Sun;

#ifdef VERTEX_PULL
// GPU-built path (render_units_gpu.rs): one draw per bucket, no vertex
// buffers. Expanded, a plain draw of soldiers * PULL_VERTS corners. Indexed
// (PULL_INDEXED), PULL_GROUP soldiers of PULL_GROUP_VERTS vertices per
// instance over the level's own index list. The soldier comes through the
// bucket's index list, the corner from the level mesh. With the pose pass
// (UNIT_POSE_READ) the list holds the soldier's pose slot, else his
// instance record, posed here.
struct PullInstance {
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

// The level mesh with its index list expanded (render_units_gpu.rs PullVertex).
struct PullVertex {
    position: vec3<f32>,
    part: f32,
    normal: vec3<f32>,
    pivot: f32,
    // unorm8 rgba.
    color: u32,
    // unorm16 each.
    atlas_uv: u32,
    // A skinned level's joint pair, a + 256 * b, and the weight of b.
    skin: u32,
    weight: f32,
};

@group(3) @binding(0) var<storage, read> pull_records: array<PullInstance>;
@group(3) @binding(1) var<storage, read> pull_index: array<u32>;
@group(3) @binding(2) var<storage, read> pull_vertices: array<PullVertex>;
// Per bucket: x = first index slot, y = corners per soldier, z = pose slots
// per soldier in the kind's pose buffer, w = soldiers per instance.
@group(3) @binding(3) var<storage, read> pull_buckets: array<vec4<u32>>;

#ifdef UNIT_BOUND_CHECK
// FL_OCCLUDE_CHECK (render_units_occlusion.rs): per kind, standing, dying,
// then settled (a corpse record, his death played out) by the way he fell
// (forward, back, either side), how far any drawn corner reached from his
// position in his facing frame: forward, back, right, left, up, down, as
// float bits.
@group(3) @binding(10) var<storage, read_write> bound_reach: array<atomic<u32>, 144>;

fn measure_reach(placed: vec3<f32>) {
    let d = placed - pose_at(P_POS).xyz;
    let yaw = pose_at(P_YAW);
    // unit_pose.wgsl `rot_y` turned his frame by cos, sin: turn it back.
    let x = d.x * yaw.x - d.z * yaw.y;
    let z = d.x * yaw.y + d.z * yaw.x;
    // A corpse record's death has played out to 1; a dying man's stops
    // a tick short of it (combat.rs freezes him at his last tick).
    // The way he fell from his topple (unit_pose.wgsl `put_common`): about
    // x forward or back by the sign of its sine, about z to either side.
    let topple = pose_at(P_DEATH);
    var state = select(0u, 1u, topple.w > 0.0);
    if topple.w > 0.999 {
        let neg = topple.y < 0.0;
        state = 2u + select(select(0u, 1u, neg), select(3u, 2u, neg), topple.z > 1.5);
    }
    let base = ((#{PULL_BUCKET}u / 4u) * 6u + state) * 6u;
    atomicMax(&bound_reach[base], bitcast<u32>(max(z, 0.0)));
    atomicMax(&bound_reach[base + 1u], bitcast<u32>(max(-z, 0.0)));
    atomicMax(&bound_reach[base + 2u], bitcast<u32>(max(x, 0.0)));
    atomicMax(&bound_reach[base + 3u], bitcast<u32>(max(-x, 0.0)));
    atomicMax(&bound_reach[base + 4u], bitcast<u32>(max(d.y, 0.0)));
    atomicMax(&bound_reach[base + 5u], bitcast<u32>(max(-d.y, 0.0)));
}
#endif

@vertex
fn vertex_pull(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
#ifdef PULL_INDEXED
    // The group's index list holds one copy of the level's list per
    // soldier, copy k offset by k times the vertex count: a corner the
    // soldier's triangles share is shaded once.
    let in_group = index / #{PULL_GROUP_VERTS}u;
    let soldier = instance * #{PULL_GROUP}u + in_group;
    let corner = index - in_group * #{PULL_GROUP_VERTS}u;
#else
    let soldier = index / #{PULL_VERTS}u;
    let corner = index - soldier * #{PULL_VERTS}u;
#endif
    let bucket = pull_buckets[#{PULL_BUCKET}u];
    let entry = pull_index[bucket.x + soldier];
#ifdef PULL_INDEXED
    if entry == 0xffffffffu {
        // Past the last soldier, in the last group (unit_build.wgsl
        // `finish_indexed`): every corner of his triangles lands on one
        // point, so they cover nothing.
        var empty: VertexOutput;
        empty.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return empty;
    }
#endif
    let v = pull_vertices[corner];
#ifdef UNIT_POSE_READ
    // The level rides the top two bits and the occlusion check's mark the
    // bit below (unit_build.wgsl `queue_draw`).
    pose_from((entry & 0x1fffffffu) * bucket.z);
#else
    let inst = pull_records[entry & 0x1fffffffu];
    pose_begin(inst.pos_scale, inst.color, inst.anim, inst.anim2, globals.time, false);
#endif
#ifdef UNIT_LIGHT_FLAT
    // The normal field holds the triangle's three corner normals
    // (render_units_gpu.rs `pack_normals`), all three turned by this
    // corner's pose in the one `place`. A part-posed level's triangle lies
    // on one part, so that is its corners' own turn; on a skinned level the
    // other two corners' joints can differ a little, and their normals take
    // this corner's turn.
    let words = bitcast<vec3<u32>>(v.normal);
    let placed = place(
        v.position,
        oct_decode(words.x),
        v.part,
        v.pivot,
        v.skin,
        v.weight,
        oct_decode(words.y),
        oct_decode(words.z),
    );
    let second = placed.second;
    let third = placed.third;
#else
    let placed = place(v.position, v.normal, v.part, v.pivot, v.skin, v.weight);
#endif
#ifdef UNIT_BOUND_CHECK
    measure_reach(placed.position);
#endif
    let team = pose_at(P_TEAM);
    var rgb = team.rgb;
#ifdef LOD_DEBUG
    // FL_LOD_DEBUG: tint by level (L1 green, L2 yellow, L3 red), the
    // level rides the top two bits of the index entry.
    let lod = entry >> 30u;
    if lod > 0u {
        var tint = vec3<f32>(1.0, 0.15, 0.1);
        if lod == 1u {
            tint = vec3<f32>(0.2, 1.0, 0.3);
        } else if lod == 2u {
            tint = vec3<f32>(1.0, 0.9, 0.1);
        }
        rgb = rgb * 0.4 + tint * 0.6;
    }
#endif
#ifdef UNIT_LIGHT_LEVEL
    // The light last level (unit_glb.rs `light_last_level`): rgb holds the
    // square root of the linear colour, a the team tint mask, which tints
    // as the atlas mask does.
    let face = unpack4x8unorm(v.color);
    let v_color = vec4<f32>(face.rgb * face.rgb * mix(vec3<f32>(1.0), rgb, face.a), 0.0);
#else
    let v_color = unpack4x8unorm(v.color);
#endif
    var out = unit_output(
        placed.position,
        placed.normal,
        v_color,
        rgb,
        pose_at(P_DEATH).w,
        team.w,
        unpack2x16unorm(v.atlas_uv),
    );
#ifdef UNIT_LIGHT_FLAT
    // The triangle's light is the mean of its three corners' lights, the
    // mean the blended corner light gives it.
    let light = (sky_and_sun(normalize(placed.normal)) + sky_and_sun(normalize(second))
        + sky_and_sun(normalize(third))) / 3.0;
    let base = mix(v_color.rgb, rgb, v_color.a);
    out.color = vec4<f32>(base * (light.x + light.y), light.y / (light.x + light.y));
#endif
#ifdef UNIT_BOUND_CHECK
    // A soldier the occlusion test hid draws flat magenta (unit_build.wgsl
    // `occlusion`).
    out.packed.y |= ((entry >> 29u) & 1u) << 8u;
#endif
    return out;
}
#else
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    pose_begin(vertex.i_pos_scale, vertex.i_color, vertex.i_anim, vertex.i_anim2, globals.time, false);
#ifdef UNIT_LIGHT_FLAT
    // A flat-lit batch drawn through `vertex_pull` gives its fragment stage
    // UNIT_LIGHT_FLAT but not VERTEX_PULL, so that stage builds this entry
    // point under UNIT_LIGHT_FLAT too; the batch never draws with it.
    let placed = place(
        vertex.position,
        vertex.normal,
        vertex.part_pivot.x,
        vertex.part_pivot.y,
        0u,
        0.0,
        vertex.normal,
        vertex.normal,
    );
#else
    let placed = place(vertex.position, vertex.normal, vertex.part_pivot.x, vertex.part_pivot.y, 0u, 0.0);
#endif
    let team = pose_at(P_TEAM);
    return unit_output(
        placed.position,
        placed.normal,
        vertex.v_color,
        team.rgb,
        pose_at(P_DEATH).w,
        team.w,
        vertex.atlas_uv,
    );
}
#endif

// The posed corner, lit, and what the fragment needs. `v_color` is the
// part material (a = team-colour blend amount), `death` the topple's
// progress, `flash` the hit flash.
fn unit_output(
    position: vec3<f32>,
    normal: vec3<f32>,
    v_color: vec4<f32>,
    team: vec3<f32>,
    death: f32,
    flash: f32,
    atlas_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    // The pose is built in world space: the bucket entity has no
    // transform of its own.
    out.clip_position = position_world_to_clip(position);
#ifdef UNIT_SHADOW_DEPTH_CLAMP
    // Shadow pass on a device without depth clip control: a soldier
    // between the sun and the cascade's near plane lands on that plane
    // instead of being clipped (render_units_shadow.rs).
    out.clip_position.z = min(out.clip_position.z, out.clip_position.w);
#endif
    // Lambert from the scene's sun, so shading and shadow agree, over a
    // hemispheric ambient brighter from above, at the corner's normal
    // turned with the pose in `place`. Most triangles of the model files
    // have different normals at their corners, so the light is blended
    // across the face, except under UNIT_LIGHT_FLAT (`vertex_pull`).
    let n = normalize(normal);
    let light = sky_and_sun(n);
    let sky = light.x;
    let direct = light.y;
    // Part material blended with the team color (a = team amount), lit.
    // The fragment takes the shadow off the sun's share, then the hit
    // flash lerps toward white and death darkens.
    let base = mix(v_color.rgb, team, v_color.a);
    out.color = vec4<f32>(base * (sky + direct), direct / (sky + direct));
    out.packed = vec2<u32>(
        pack4x8unorm(vec4<f32>(team, death)),
        pack4x8unorm(vec4<f32>(flash, 0.0, 0.0, 0.0)),
    );
#ifdef UNIT_SHADOW_RECEIVE
    out.shadow_position = position + n * shadow_normal_bias(position);
#endif
#ifdef UNIT_ATLAS
    out.atlas_uv = atlas_uv;
#endif
    return out;
}

// The light at a unit normal: x the hemispheric sky, brighter from above,
// y the sun's Lambert term.
fn sky_and_sun(n: vec3<f32>) -> vec2<f32> {
    let sky = 0.30 + 0.20 * (0.5 + 0.5 * n.y);
    let direct = 0.65 * max(dot(n, sun.direction_to_light.xyz), 0.0);
    return vec2<f32>(sky, direct);
}

// A unit direction from its octahedral word: x and y of the projection as
// unorm16 (render_units_gpu.rs `oct_encode`).
fn oct_decode(word: u32) -> vec3<f32> {
    let p = unpack2x16unorm(word) * 2.0 - 1.0;
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// Depth along the camera's view axis, negative in front of it.
fn view_depth(world_position: vec3<f32>) -> f32 {
    return dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z,
    ), vec4<f32>(world_position, 1.0));
}

// How far off the surface the sun's shadow is sampled, in metres: the
// light's normal bias scaled to the texel of the cascade this depth
// reads, as bevy_pbr::shadows::sample_directional_cascade does per pixel
// from the normal. The fragment no longer carries the normal, so the
// offset is applied here. In a blend band the next cascade is read too,
// with a larger texel: its bias serves both, a little more than the
// first needs. 0 with shadows off.
fn shadow_normal_bias(world_position: vec3<f32>) -> f32 {
    if sun.params.x == 0.0 {
        return 0.0;
    }
    let depth = -view_depth(world_position);
    let n_cascades = u32(sun.params.w);
    // bevy_pbr::shadows::get_cascade_index: the first cascade whose far
    // bound is past this depth.
    var c = n_cascades;
    for (var i = 0u; i < n_cascades; i++) {
        if depth < sun.far_bounds[i] {
            c = i;
            break;
        }
    }
    if c >= n_cascades {
        return 0.0;
    }
    if c + 1u < n_cascades && depth >= (1.0 - sun.params.z) * sun.far_bounds[c] {
        c += 1u;
    }
    return sun.params.y * sun.texel_sizes[c];
}

// How much of the scene's sun reaches this point, 0..1: the first
// directional light's shadow cascades at a position already pushed off
// the surface by the normal bias, so no normal goes in. 1 past the last
// cascade or with shadows off (FL_SHADOWS=0).
fn sun_shadow(shadow_position: vec3<f32>, frag_xy: vec2<f32>) -> f32 {
    if (lights.directional_lights[0].flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) == 0u {
        return 1.0;
    }
    let p = vec4<f32>(shadow_position, 1.0);
    return fetch_directional_shadow(0u, p, vec3<f32>(0.0), view_depth(shadow_position), frag_xy);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
#ifdef UNIT_BOUND_CHECK
    if (in.packed.y & 256u) != 0u {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }
#endif
    // rgb = team colour, a = death darkening.
    let team = unpack4x8unorm(in.packed.x);
    let flash = unpack4x8unorm(in.packed.y).x;
#ifdef UNIT_ATLAS
    // The atlas mask tints rather than replaces, so cloth keeps its weave
    // under the team colour.
    let texel = textureSample(unit_atlas, unit_atlas_sampler, in.atlas_uv);
    let tint = mix(vec3<f32>(1.0), team.rgb, texel.a);
    var rgb = in.color.rgb * texel.rgb * tint;
#else
    var rgb = in.color.rgb;
#endif
#ifdef UNIT_SHADOW_RECEIVE
    // The sun's share of the light, less what its shadow takes. A face
    // turned from the sun has no share and skips the lookup.
    if in.color.a > 0.0 {
        rgb *= 1.0 - in.color.a * (1.0 - sun_shadow(in.shadow_position, in.clip_position.xy));
    }
#endif
    rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), flash * 0.8);
    rgb = rgb * (1.0 - 0.45 * team.a);
    return vec4<f32>(rgb, 1.0);
}
