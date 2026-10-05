# Kintsugi CR7

An interactive, cinematic generative-art tribute to **Cristiano Ronaldo**.

The figure is a stylized silhouette built procedurally from hundreds of triangular stained-glass mosaic shards. As you progress through his life's 28 chapters across 7 acts, doubt, loss, injury, and criticism shatter the silhouette, while comebacks and achievements repair it with glowing gold kintsugi seams.

## Features
- **Pure Generative Art**: No photos, video playback, or external image assets in runtime. Everything is rendered via HTML5 Canvas and mathematical triangulation.
- **Dynamic Physics & Skinning**: Interactive Delaunay shard physics with mouse/touch repulsion, spring damping, and dynamic fracture mechanics.
- **Pose Extraction & Track Runner**: Build-time pose extractor using PyTorch keypoint estimation, normalized to 15-joint skeletal keyframes with Catmull-Rom spline interpolation and hang-time deceleration.
- **Kintsugi Gold Seams**: Procedural gold filling on fracture edges with adaptive bloom lighting and resonance audio synthesis.
- **Interactive Chapter System & Audio**: 28 historical beats covering Sporting CP, Manchester United, Real Madrid, Juventus, Al-Nassr, and Portugal with procedural Web Audio soundscapes.

## Running Locally
Simply open `index.html` in any modern web browser or serve with a local static server:

```bash
# Python
python3 -m http.server 8000

# or Node
npx serve .
```

## Controls
- `Space`: Play / Pause auto-progression
- `Right Arrow` / `Left Arrow`: Next / Previous Chapter
- `Click + Drag`: Scatter & interact with mosaic shards
- `Click + Hold`: Gather golden energy and trigger gold shockwave
- `S`: Trigger manual fracture / repair cycle
- `D`: Toggle Debug HUD (wireframe skeleton, joint coordinates & track timeline scrubber)
- `E`: Export current interpolated pose coordinates as JSON
