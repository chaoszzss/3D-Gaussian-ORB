// Tiny synthetic SH0 shader test, called after real-model captures while locked.
export async function directionGpuSmoke(page) {
  return page.evaluate(async () => {
    const THREE = await import('/vendor/three/build/three.module.js');
    const { SparkRenderer, SplatMesh, ExtSplats, dyno } = await import('/vendor/spark.js');
    const { attachSurfaceModifiers, sha256 } = await import('/gaussian_direction_support.js');
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(256, 256); renderer.setClearColor(0, 1);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, .2, 100);
    let splats = new ExtSplats({ maxSplats: 3 }); await splats.initialized;
    const centers = [[0, 0, 0], [-.65, 0, 0], [.65, 0, 0]];
    const colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]], flags = Uint8Array.from([1, 0, 3]);
    centers.forEach((p, i) => splats.pushSplat(new THREE.Vector3(...p), new THREE.Vector3(.15, .15, .015),
      new THREE.Quaternion(), .9, new THREE.Color(...colors[i])));
    // Spark pushSplat fills CPU arrays but does not materialize the source GPU
    // textures. Reinitialize through the public extArrays constructor, just as
    // the actual PLY worker decoder does; otherwise the fixture renders empty.
    splats = new ExtSplats({ extArrays: splats.extArrays, numSplats: 3 });
    await splats.initialized;
    const mesh = new SplatMesh({ extSplats: splats, enableLod: false, editable: false, raycastable: false });
    await mesh.initialized;
    const header = new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement vertex 3\n' +
      ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3'].map(f => `property float ${f}\n`).join('') + 'end_header\n');
    const ply = new Uint8Array(header.length + 3 * 7 * 4); ply.set(header);
    const view = new DataView(ply.buffer);
    centers.forEach((p, i) => [...p, 1, 0, 0, 0].forEach((v, j) => view.setFloat32(header.length + (7 * i + j) * 4, v, true)));
    const metadata = { schema_version: 1, ply: { sha256: await sha256(ply), gaussians: 3 },
      side_flags: { sha256: await sha256(flags), dtype: 'uint8', count: 3 },
      normal_convention: 'quaternion_local_z', observed_cosine_min: .1, render_grazing_cosine: .05,
      policy: 'strong_unilateral_train_group_support', training_evidence: {
        synthetic_unit_fixture: true, split: 'train', train_only: true, heldout_used: false,
        minimum_unique_groups: 3, opposite_side_required_count: 0 } };
    const support = { metadata, flags, plySha256: metadata.ply.sha256,
      metadataSha256: await sha256(new TextEncoder().encode(JSON.stringify(metadata))) };
    const attachment = attachSurfaceModifiers({ THREE, dyno, mesh, support, plyBytes: ply,
      maxTextureSize: renderer.capabilities.maxTextureSize });
    const spark = new SparkRenderer({ renderer, enable2DGS: true, sortRadial: false, accumExtSplats: true,
      enableLod: false, maxStdDev: 3, minAlpha: 1 / 255 });
    scene.add(mesh, spark);
    const rows = [];
    try {
      for (const z of [2, -2]) {
        camera.position.set(0, 0, z); camera.lookAt(0, 0, 0); camera.updateMatrixWorld(true);
        for (let i = 0; i < 12; i++) {
          renderer.render(scene, camera); await new Promise(resolve => setTimeout(resolve, 50));
        }
        if (spark.sorting || spark.sortDirty) throw new Error('Synthetic direction sort did not settle');
        renderer.render(scene, camera);
        const gl = renderer.getContext(), rgba = new Uint8Array(256 * 256 * 4);
        gl.readPixels(0, 0, 256, 256, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
        const pixels = centers.map(p => {
          const clip = new THREE.Vector3(...p).project(camera);
          const x = Math.floor((clip.x + 1) * 128), y = Math.floor((clip.y + 1) * 128);
          return [...rgba.subarray(4 * (y * 256 + x), 4 * (y * 256 + x) + 4)];
        });
        rows.push({ camera_z: z, pixels, png: renderer.domElement.toDataURL('image/png') });
      }
      const passed = rows[0].pixels[0][0] > 180 && rows[1].pixels[0][0] < 5 &&
        rows.every(row => row.pixels[1][2] > 180 && row.pixels[2][1] > 180);
      return { status: passed ? 'passed' : 'failed', dataset_kind: 'synthetic_unit_test',
        gaussian_count: 3, sh_degree: 0, flags: [...flags],
        semantics: 'red positive-only, blue unknown, green both; back view hides only red', rows };
    } finally {
      attachment.dispose(); mesh.dispose(); spark.dispose(); renderer.dispose();
    }
  });
}
