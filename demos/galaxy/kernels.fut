-- N-body gravity. `step` advances the whole system one timestep, O(n^2). Pairwise
-- force accumulation plus a leapfrog-style integration. Position and velocity
-- travel as a tuple in and a tuple out. Softening avoids the r->0 singularity
-- (self-term contributes 0).

let accel [n] (pos: [n][3]f32) (mass: [n]f32) (soft: f32) : [n][3]f32 =
  map (\pa ->
        reduce (\u v -> map2 (+) u v) [0f32, 0f32, 0f32]
               (map2 (\pb mb ->
                        let dx = pb[0] - pa[0]
                        let dy = pb[1] - pa[1]
                        let dz = pb[2] - pa[2]
                        let r2 = dx * dx + dy * dy + dz * dz + soft * soft
                        let inv = 1f32 / (r2 * f32.sqrt r2)
                        in [mb * dx * inv, mb * dy * inv, mb * dz * inv])
                     pos mass))
      pos

entry step [n] (pos: [n][3]f32) (vel: [n][3]f32) (mass: [n]f32)
               (dt: f32) (soft: f32) : ([n][3]f32, [n][3]f32) =
  let acc = accel pos mass soft
  let vel' = map2 (\v a -> map2 (\vi ai -> vi + ai * dt) v a) vel acc
  let pos' = map2 (\p v -> map2 (\pi vi -> pi + vi * dt) p v) pos vel'
  in (pos', vel')
