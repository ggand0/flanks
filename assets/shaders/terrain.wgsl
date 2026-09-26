#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_view_bindings::view,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var pasture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var ground_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var pasture_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var stone: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var stone_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var earth: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var earth_normal: texture_2d<f32>;

fn hash_ground(p: vec2<f32>) -> f32 {
    var q = fract(vec3<f32>(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn ground_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let blend = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash_ground(cell), hash_ground(cell + vec2<f32>(1.0, 0.0)), blend.x),
        mix(hash_ground(cell + vec2<f32>(0.0, 1.0)), hash_ground(cell + 1.0), blend.x),
        blend.y,
    );
}

struct PastureSample {
    color: vec3<f32>,
    normal_roughness: vec4<f32>,
};

fn pasture_patch(uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>, cell: vec2<f32>, detail: f32) -> PastureSample {
    let turn = u32(hash_ground(cell + 61.0) * 4.0);
    let c = select(select(-1.0, 1.0, turn == 0u), 0.0, (turn & 1u) == 1u);
    let s = select(select(-1.0, 1.0, turn == 1u), 0.0, (turn & 1u) == 0u);
    let rotation = mat2x2<f32>(vec2<f32>(c, s), vec2<f32>(-s, c));
    let offset = vec2<f32>(hash_ground(cell), hash_ground(cell + 37.0));
    let patch_uv = rotation * uv + offset;
    var result: PastureSample;
    result.color = textureSampleGrad(pasture, ground_sampler, patch_uv, rotation * uv_dx, rotation * uv_dy).rgb;
    result.normal_roughness = vec4<f32>(0.5, 0.5, 1.0, 0.95);
    if detail > 0.001 {
        result.normal_roughness = textureSampleGrad(pasture_normal, ground_sampler, patch_uv, rotation * uv_dx, rotation * uv_dy);
        // OpenGL tangent Y points opposite texture V, so rotate XY with the UV basis.
        result.normal_roughness = vec4<f32>(
            rotation * (result.normal_roughness.xy * 2.0 - 1.0) * 0.5 + 0.5,
            result.normal_roughness.zw,
        );
    }
    return result;
}

fn sample_pasture(p: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>, detail: f32) -> PastureSample {
    // Barycentric patches share samples at their edges, hiding tile boundaries.
    let grid = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(-0.57735027, 1.15470054)) * (p / 10.0);
    let base = floor(grid);
    let f = fract(grid);
    var a = base;
    var b = base + vec2<f32>(1.0, 0.0);
    var c = base + vec2<f32>(0.0, 1.0);
    var weights = vec3<f32>(1.0 - f.x - f.y, f.x, f.y);
    if f.x + f.y > 1.0 {
        a = base + 1.0;
        b = base + vec2<f32>(0.0, 1.0);
        c = base + vec2<f32>(1.0, 0.0);
        weights = vec3<f32>(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
    }
    let sa = pasture_patch(p / 6.0, uv_dx, uv_dy, a, detail);
    let sb = pasture_patch(p / 6.0, uv_dx, uv_dy, b, detail);
    let sc = pasture_patch(p / 6.0, uv_dx, uv_dy, c, detail);
    var result: PastureSample;
    result.color = sa.color * weights.x + sb.color * weights.y + sc.color * weights.z;
    result.normal_roughness = sa.normal_roughness * weights.x + sb.normal_roughness * weights.y + sc.normal_roughness * weights.z;
    return result;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    let p = in.world_position.xyz;
    let n = normalize(in.world_normal);
    let distance_to_camera = distance(p, view.world_position);
    let detail = 1.0 - smoothstep(50.0, 220.0, distance_to_camera);

    // World-space mapping remains continuous when a chunk is rebuilt.
    let dx = dpdx(p);
    let dy = dpdy(p);
    let grass = sample_pasture(p.xz, dx.xz / 6.0, dy.xz / 6.0, detail);
    let soil_uv = p.xz / 3.5;
    let stone_uv = p.xz / 4.0;
    let soil_color = textureSample(earth, ground_sampler, soil_uv).rgb;
    var rock_color = textureSample(stone, ground_sampler, stone_uv).rgb;
    var soil_nr = vec4<f32>(0.5, 0.5, 1.0, 0.95);
    var rock_nr = soil_nr;
    if detail > 0.001 {
        soil_nr = textureSampleGrad(earth_normal, ground_sampler, soil_uv, dx.xz / 3.5, dy.xz / 3.5);
        rock_nr = textureSampleGrad(stone_normal, ground_sampler, stone_uv, dx.xz / 4.0, dy.xz / 4.0);
    }

    // Side projections avoid stretching the stony layer on steep banks.
    let side_weight = smoothstep(0.25, 0.75, 1.0 - n.y);
    if side_weight > 0.001 {
        let side_x = textureSampleGrad(stone, ground_sampler, p.zy / 4.0, dx.zy / 4.0, dy.zy / 4.0).rgb;
        let side_z = textureSampleGrad(stone, ground_sampler, p.xy / 4.0, dx.xy / 4.0, dy.xy / 4.0).rgb;
        let side_color = mix(side_z, side_x, abs(n.x) / max(abs(n.x) + abs(n.z), 0.001));
        rock_color = mix(rock_color, side_color, side_weight);
        // Fade tangent detail where the projection changes basis.
        rock_nr = mix(rock_nr, vec4<f32>(0.5, 0.5, 1.0, 0.95), side_weight);
    }

    let broad = ground_noise(p.xz / 120.0 + 17.0);
    let patches = ground_noise(p.xz / 19.0 + 43.0);
    let fine = ground_noise(p.xz / 2.5);
    let dry = clamp(in.color.r + (patches - 0.5) * 0.38, 0.0, 1.0);
    let soil = smoothstep(0.05, 0.95, clamp(in.color.g + (fine - 0.5) * 0.13, 0.0, 1.0));
    let rock = smoothstep(0.05, 0.95, in.color.b) * (1.0 - soil);
    let damp = in.color.a;

    let grass_tint = mix(vec3<f32>(0.83, 1.12, 0.72), vec3<f32>(1.30, 1.15, 0.76), dry);
    var color = grass.color * grass_tint;
    color = mix(color, soil_color * vec3<f32>(0.85, 0.89, 0.82), soil);
    color = mix(color, rock_color * vec3<f32>(0.92, 0.95, 0.96), rock);
    color *= (0.82 + 0.30 * broad + 0.10 * patches) * (1.0 - damp * 0.24);

    var pbr = pbr_input_from_standard_material(in, is_front);
    pbr.material.base_color = vec4<f32>(color, 1.0);
    let nr = mix(mix(grass.normal_roughness, soil_nr, soil), rock_nr, rock);
    pbr.material.perceptual_roughness = mix(0.95, clamp(nr.a, 0.78, 1.0), detail);
    // Texture V increases along +Z; the OpenGL normal's +Y points toward -Z.
    let tangent = normalize(vec3<f32>(n.y, -n.x, 0.0));
    let bitangent = cross(n, tangent);
    let detail_normal = nr.xyz * 2.0 - 1.0;
    pbr.N = normalize(n + (tangent * detail_normal.x + bitangent * detail_normal.y) * detail * 0.32);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
}
