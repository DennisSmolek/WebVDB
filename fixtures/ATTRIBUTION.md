# Fixture attribution and usage terms

Binary fixtures remain git-ignored and are not included in npm packages. This
tracked record preserves the source and usage terms needed to reproduce local
fixture sets.

## JangaFX EmberGen free VDB packs

- Creator: JangaFX Software, <https://jangafx.com>
- Source: <https://jangafx.com/software/embergen/download/free-vdb-animations>
- Terms supplied with the downloaded packs: Creative Commons CC0 1.0 Universal,
  <https://creativecommons.org/publicdomain/zero/1.0/>

JangaFX's bundled README states that the VDBs were generated with EmberGen, may
be used however the recipient chooses, and are licensed under CC0. The bundled
CC0 text waives copyright and related rights to the greatest extent permitted,
provides the standard fallback license where waiver is ineffective, grants no
trademark or patent rights, and supplies the work without warranties.

Each downloaded pack's original `LICENSE.txt` and `README.txt` stays beside its
ignored binaries locally. Credit: **VDBs created by JangaFX Software using
EmberGen**.

The Fire pack used for temporal playback contains an actual numbered sequence.
The Cloud pack contains independent `_0000` variants and must not be described
or baked as an animation.

## OpenVDB sample models

The pinned `sphere.vdb`, `cube.vdb`, `smoke.vdb`, and `utahteapot.vdb` fixtures
come from the OpenVDB sample-model collection and are mirrored at commit
`2173458e3b6d646deee5401ec6493079b09c091c` of
<https://github.com/mjurczyk/openvdb>. The hosting repository is MIT licensed;
the sample assets retain the OpenVDB download page's CC BY-SA 4.0 terms:
<https://www.openvdb.org/download/>.

## Stanford bunny derivatives

`bunny.vdb` and `bunny_cloud.vdb` are volumetric derivatives of the Stanford
Bunny from the Stanford University Computer Graphics Laboratory. Stanford
permits research use and free mirroring/redistribution with attribution, but
prohibits commercial use or inclusion in a product for sale without permission:
<https://graphics.stanford.edu/data/3Dscanrep/>.

Those restrictions are not equivalent to CC BY-SA 4.0. Bunny derivatives are
therefore excluded from the automated fixture fetch, repository, parity corpus,
and all npm package tarballs.

## Walt Disney Animation Studios cloud

The WDAS cloud data set is © Walt Disney Animation Studios and distributed
under CC BY-SA 3.0:
<https://disneyanimation.com/resources/clouds/>.
