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
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var<uniform> natural_ground: u32;

fn hash_ground(p: vec2<f32>) -> f32 {
    var q = fract(vec3<f32>(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn cover_variation(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let weight = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash_ground(cell), hash_ground(cell + vec2<f32>(1.0, 0.0)), weight.x),
        mix(hash_ground(cell + vec2<f32>(0.0, 1.0)), hash_ground(cell + 1.0), weight.x),
        weight.y,
    ) * 2.0 - 1.0;
}

fn meadow_transition(field: vec4<f32>, uv: vec2<f32>, p: vec2<f32>, footprint: f32) -> vec4<f32> {
    // This corridor belongs to the authored eastern dry meadow. Keep the
    // pasture and the outer dry ground outside it unchanged.
    let corridor = smoothstep(0.57, 0.64, uv.x) * (1.0 - smoothstep(0.86, 0.94, uv.x));
    if corridor < 0.001 {
        return field;
    }

    // Estimate grass and dry-ground coverage over metres, rather than retaining
    // the source image's small islands as enlarged, hard-edged color outlines.
    let texel_m = coverage_bounds.z / f32(textureDimensions(coverage).x);
    let mip = log2(max(8.0, footprint) / texel_m);
    var grass_sum = vec4<f32>(0.0);
    var dry_sum = vec4<f32>(0.0);
    var alpha_sum = 0.0;
    var local_mean = field.rgb;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let weight = select(1.0, 2.0, x == 0) * select(1.0, 2.0, y == 0);
            let offset = vec2<f32>(f32(x), f32(y)) * 16.0 / coverage_bounds.zw;
            let sample = textureSampleLevel(coverage, coverage_sampler, uv + offset, mip);
            if x == 0 && y == 0 {
                local_mean = sample.rgb;
            }
            let warmth = (sample.r - sample.g) / max(sample.r + sample.g, 0.001);
            let dry = smoothstep(-0.035, 0.075, warmth);
            grass_sum += vec4<f32>(sample.rgb, 1.0) * ((1.0 - dry) * weight);
            dry_sum += vec4<f32>(sample.rgb, 1.0) * (dry * weight);
            alpha_sum += sample.a * weight;
        }
    }
    let coverage_weight = dry_sum.a / 16.0;
    let edge = smoothstep(0.04, 0.22, coverage_weight) * (1.0 - smoothstep(0.78, 0.96, coverage_weight));
    if edge < 0.001 {
        return field;
    }
    let grass_color = grass_sum.rgb / max(grass_sum.a, 0.001);
    let dry_color = dry_sum.rgb / max(dry_sum.a, 0.001);
    // Small coverage interruptions have explicit world sizes. Filter them out
    // when unresolved, and confine them to the mixed vegetation corridor.
    let breakup = cover_variation(p / vec2<f32>(7.0, 4.0)) * 0.16 * (1.0 - smoothstep(3.0, 8.0, footprint))
        + cover_variation(p / 2.5 + 37.0) * 0.08 * (1.0 - smoothstep(1.0, 3.0, footprint));
    let dry = clamp(coverage_weight + breakup * 4.0 * coverage_weight * (1.0 - coverage_weight), 0.0, 1.0);
    // Preserve resolved surface grain while changing the larger material mix.
    let luminance = vec3<f32>(0.2126, 0.7152, 0.0722);
    let grain = clamp(dot(field.rgb, luminance) / max(dot(local_mean, luminance), 0.001), 0.65, 1.45);
    let transition = vec4<f32>(mix(grass_color, dry_color, dry) * grain, alpha_sum / 16.0);
    return mix(field, transition, corridor * edge);
}

struct GroundSample {
    color: vec3<f32>,
    normal_roughness: vec4<f32>,
};

fn ground_patch(
    color_map: texture_2d<f32>, normal_map: texture_2d<f32>,
    uv: vec2<f32>, uv_dx: vec2<f32>, uv_dy: vec2<f32>,
    cell: vec2<f32>, detail: f32,
) -> GroundSample {
    let angle = hash_ground(cell + 61.0) * 6.2831853;
    let c = cos(angle);
    let s = sin(angle);
    let rotation = mat2x2<f32>(vec2<f32>(c, s), vec2<f32>(-s, c));
    let offset = vec2<f32>(hash_ground(cell), hash_ground(cell + 37.0));
    let patch_uv = rotation * uv + offset;
    var result: GroundSample;
    result.color = textureSampleGrad(color_map, ground_sampler, patch_uv, rotation * uv_dx, rotation * uv_dy).rgb;
    result.normal_roughness = vec4<f32>(0.5, 0.5, 1.0, 0.95);
    if detail > 0.001 {
        result.normal_roughness = textureSampleGrad(normal_map, ground_sampler, patch_uv, rotation * uv_dx, rotation * uv_dy);
        // OpenGL tangent Y points opposite texture V, so rotate XY with the UV basis.
        result.normal_roughness = vec4<f32>(
            rotation * (result.normal_roughness.xy * 2.0 - 1.0) * 0.5 + 0.5,
            result.normal_roughness.zw,
        );
    }
    return result;
}

fn sample_ground(
    color_map: texture_2d<f32>, normal_map: texture_2d<f32>, mean: vec3<f32>,
    p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, tile_m: f32, detail: f32,
) -> GroundSample {
    // Normalize the residual variance so blended regions do not become softer
    // than patch centers. The linear mean keeps the distant color stable.
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
    let sa = ground_patch(color_map, normal_map, p / tile_m, uv_dx, uv_dy, a, detail);
    let sb = ground_patch(color_map, normal_map, p / tile_m, uv_dx, uv_dy, b, detail);
    let sc = ground_patch(color_map, normal_map, p / tile_m, uv_dx, uv_dy, c, detail);
    var result: GroundSample;
    let residual = (sa.color - mean) * weights.x + (sb.color - mean) * weights.y + (sc.color - mean) * weights.z;
    result.color = max(vec3<f32>(0.0), mean + residual * inverseSqrt(dot(weights, weights)));
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
    // Mip filtering removes subpixel detail continuously, without a separate
    // distance band that turns textured ground into a flat color.
    let grass = sample_ground(
        pasture, pasture_normal, select(vec3<f32>(0.5), vec3<f32>(0.158131, 0.119077, 0.049214), natural_ground != 0u),
        p.xz, dx.xz, dy.xz, 2.51, detail,
    );
    let soil_sample = sample_ground(
        earth, earth_normal, vec3<f32>(0.194145, 0.120636, 0.057649),
        p.xz + 83.0, dx.xz, dy.xz, 3.5, detail,
    );
    let soil_color = soil_sample.color;
    let soil_nr = soil_sample.normal_roughness;
    var rock_color = vec3<f32>(0.284067, 0.232351, 0.136810);
    var rock_nr = vec4<f32>(0.5, 0.5, 1.0, 0.95);
    if natural_ground == 0u && in.color.b > 0.05 {
        let rock_sample = sample_ground(
            stone, stone_normal, rock_color,
            p.xz - 157.0, dx.xz, dy.xz, 4.0, detail,
        );
        rock_color = rock_sample.color;
        rock_nr = rock_sample.normal_roughness;
    }

    // Side projections avoid stretching the stony layer on steep banks.
    let side_weight = smoothstep(0.25, 0.75, 1.0 - n.y);
    if natural_ground == 0u && side_weight > 0.001 {
        let side_x = textureSampleGrad(stone, ground_sampler, p.zy / 4.0, dx.zy / 4.0, dy.zy / 4.0).rgb;
        let side_z = textureSampleGrad(stone, ground_sampler, p.xy / 4.0, dx.xy / 4.0, dy.xy / 4.0).rgb;
        let side_color = mix(side_z, side_x, abs(n.x) / max(abs(n.x) + abs(n.z), 0.001));
        rock_color = mix(rock_color, side_color, side_weight);
        // Fade tangent detail where the projection changes basis.
        rock_nr = mix(rock_nr, vec4<f32>(0.5, 0.5, 1.0, 0.95), side_weight);
    }

    let coverage_uv = (p.xz - coverage_bounds.xy) / coverage_bounds.zw;
    var field = textureSample(coverage, coverage_sampler, coverage_uv);
    if natural_ground != 0u {
        field = meadow_transition(field, coverage_uv, p.xz, max(length(dx.xz), length(dy.xz)));
    }
    var soil: f32;
    var rock = 0.0;
    var color: vec3<f32>;
    let soil_surface = soil_color * vec3<f32>(0.85, 0.89, 0.82);
    if natural_ground != 0u {
        // The unique albedo carries metre-scale surface structure. Native-scale
        // scan detail converges to one under mip filtering, preserving that layout.
        let grass_residual = grass.color / vec3<f32>(0.158131, 0.119077, 0.049214);
        let earth_residual = soil_color / vec3<f32>(0.194145, 0.120636, 0.057649);
        let surface_detail = clamp(mix(grass_residual, earth_residual, field.a), vec3<f32>(0.35), vec3<f32>(2.2));
        color = field.rgb * surface_detail;
        color = mix(color, soil_surface, in.color.g);
        soil = max(field.a, in.color.g);
    } else {
        // The river material retains its procedural coverage and bank layers.
        let edge_detail = (grass.color.r - 0.5) * field.g * (1.0 - field.g) * 2.0;
        soil = max(clamp(field.g + edge_detail, 0.0, 1.0), in.color.g);
        rock = smoothstep(0.05, 0.95, in.color.b) * (1.0 - soil);
        let grass_color = mix(vec3<f32>(0.075, 0.105, 0.033), vec3<f32>(0.185, 0.154, 0.073), field.r);
        let grass_detail = clamp(1.0 + (grass.color - 0.5) * 2.2, vec3<f32>(0.45), vec3<f32>(1.6));
        let stone_surface = rock_color * vec3<f32>(0.92, 0.95, 0.96);
        color = mix(grass_color * grass_detail, soil_surface, soil);
        color = mix(color, stone_surface, rock);
        color *= (0.82 + 0.36 * field.b) * (1.0 - in.color.a * 0.24);
    }

    var pbr = pbr_input_from_standard_material(in, is_front);
    pbr.material.base_color = vec4<f32>(color, 1.0);
    let nr = mix(mix(grass.normal_roughness, soil_nr, soil), rock_nr, rock);
    pbr.material.perceptual_roughness = mix(0.95, clamp(nr.a, 0.78, 1.0), detail);
    // Texture V increases along +Z; the OpenGL normal's +Y points toward -Z.
    let tangent = normalize(vec3<f32>(n.y, -n.x, 0.0));
    let bitangent = cross(n, tangent);
    let grass_normal = (grass.normal_roughness.xyz * 2.0 - 1.0) * vec3<f32>(0.18, 0.18, 1.0);
    let soil_normal = (soil_nr.xyz * 2.0 - 1.0) * vec3<f32>(0.32, 0.32, 1.0);
    let rock_normal = (rock_nr.xyz * 2.0 - 1.0) * vec3<f32>(0.32, 0.32, 1.0);
    let detail_normal = mix(mix(grass_normal, soil_normal, soil), rock_normal, rock);
    pbr.N = normalize(n + (tangent * detail_normal.x + bitangent * detail_normal.y) * detail);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
}
