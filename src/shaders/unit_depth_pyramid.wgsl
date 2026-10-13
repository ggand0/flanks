// The depth pyramid the unit build pass tests soldiers against
// (render_units_occlusion.rs). After the opaque pass, `first` takes the
// view's depth buffer to level 0, one texel per 2x2 pixels, and `down`
// halves each level into the next. A texel holds the farthest depth of
// every sample under it: the smallest value, the depth being reversed.
// Texel (x, y) of level k covers pixels [x, x + 1) * 2^(k + 1) on each
// axis. Level 0 rounds the screen's size up, so its last texel may read
// one pixel row or column only. Each later level halves the one before
// rounding down, as mip levels do, so the last texel of a level whose
// source is odd also takes the source's last row or column: the last texel
// covers everything to the screen's edge (unit_build.wgsl `occluded`).

#ifdef MULTISAMPLE
@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var depth: texture_depth_2d;
#endif
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(dst)) {
        return;
    }
    let size = textureDimensions(depth);
    var farthest = 1.0;
    for (var k = 0u; k < 4u; k++) {
        let p = id.xy * 2u + vec2<u32>(k & 1u, k >> 1u);
        if all(p < size) {
#ifdef MULTISAMPLE
            for (var s = 0u; s < textureNumSamples(depth); s++) {
                farthest = min(farthest, textureLoad(depth, p, i32(s)));
            }
#else
            farthest = min(farthest, textureLoad(depth, p, 0));
#endif
        }
    }
    textureStore(dst, id.xy, vec4<f32>(farthest));
}

@compute @workgroup_size(8, 8, 1)
fn down(@builtin(global_invocation_id) id: vec3<u32>) {
    let out = textureDimensions(dst);
    if any(id.xy >= out) {
        return;
    }
    let size = textureDimensions(src);
    // Two source texels per axis, three for the last texel.
    let span = select(vec2<u32>(2u), vec2<u32>(3u), id.xy == out - 1u);
    var farthest = 1.0;
    for (var y = 0u; y < span.y; y++) {
        for (var x = 0u; x < span.x; x++) {
            let p = id.xy * 2u + vec2<u32>(x, y);
            if all(p < size) {
                farthest = min(farthest, textureLoad(src, p, 0).r);
            }
        }
    }
    textureStore(dst, id.xy, vec4<f32>(farthest));
}
