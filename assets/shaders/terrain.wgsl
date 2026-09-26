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
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var coverage: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var coverage_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var<uniform> coverage_bounds: vec4<f32>;

fn hash_ground(p: vec2<f32>) -> f32 {
    var q = fract(vec3<f32>(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

struct PastureSample {
    color: vec3<f32>,
    normal_roughness: vec4<f32>,
};

fn pasture_patch(uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>, cell: vec2<f32>, detail: f32) -> PastureSample {
    let angle = hash_ground(cell + 61.0) * 6.2831853;
    let c = cos(angle);
    let s = sin(angle);
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

fn sample_pasture(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, tile_m: f32, detail: f32) -> PastureSample {
    // Normalize the residual variance so blended regions do not become softer
    // than patch centers. The prepared texture has a linear mean of 0.5.
    let uv_dx = dx / tile_m;
    let uv_dy = dy / tile_m;
    let grid = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(-0.57735027, 1.15470054)) * (p / (tile_m * 1.7));
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
    let sa = pasture_patch(p / tile_m, uv_dx, uv_dy, a, detail);
    let sb = pasture_patch(p / tile_m, uv_dx, uv_dy, b, detail);
    let sc = pasture_patch(p / tile_m, uv_dx, uv_dy, c, detail);
    var result: PastureSample;
    let residual = (sa.color - 0.5) * weights.x + (sb.color - 0.5) * weights.y + (sc.color - 0.5) * weights.z;
    result.color = 0.5 + residual * inverseSqrt(dot(weights, weights));
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
    // A projected footprint also fades detail on distant, grazing surfaces.
    let footprint = max(length(dx.xz), length(dy.xz));
    let color_detail = (1.0 - smoothstep(65.0, 300.0, distance_to_camera))
        * (1.0 - smoothstep(0.15, 0.7, footprint));
    var grass: PastureSample;
    grass.color = vec3<f32>(0.5);
    grass.normal_roughness = vec4<f32>(0.5, 0.5, 1.0, 0.95);
    if color_detail > 0.001 || detail > 0.001 {
        grass = sample_pasture(p.xz, dx.xz, dy.xz, 6.0, detail);
    }
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

    let field = textureSample(coverage, coverage_sampler, (p.xz - coverage_bounds.xy) / coverage_bounds.zw);
    let dry = field.r;
    let soil = max(field.g, in.color.g);
    let rock = smoothstep(0.05, 0.95, in.color.b) * (1.0 - soil);
    let damp = in.color.a;

    // Meter-scale turf variation bridges the coverage field and blade detail.
    // Its footprint fade leaves only the broad field when it becomes subpixel.
    let turf = sample_pasture(p.xz + vec2<f32>(173.0, -291.0), dx.xz, dy.xz, 34.0, 0.0);
    let turf_amount = 1.0 - smoothstep(0.8, 3.0, footprint);
    let turf_variation = clamp(1.0 + (turf.color.r - 0.5) * 1.9, 0.55, 1.5);
    let grass_color = mix(vec3<f32>(0.085, 0.115, 0.042), vec3<f32>(0.175, 0.153, 0.078), dry);
    let grass_detail = clamp(1.0 + (grass.color - 0.5) * 1.6, vec3<f32>(0.45), vec3<f32>(1.6));
    var color = grass_color * mix(vec3<f32>(1.0), grass_detail, color_detail)
        * mix(1.0, turf_variation, turf_amount);
    let soil_surface = mix(vec3<f32>(0.125, 0.101, 0.070), soil_color * vec3<f32>(0.85, 0.89, 0.82), color_detail);
    let stone_surface = mix(vec3<f32>(0.18, 0.174, 0.15), rock_color * vec3<f32>(0.92, 0.95, 0.96), color_detail);
    color = mix(color, soil_surface, soil);
    color = mix(color, stone_surface, rock);
    color *= (0.82 + 0.36 * field.b) * (1.0 - damp * 0.24);

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
