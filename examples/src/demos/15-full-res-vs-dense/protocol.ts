/**
 * Demo 15: the assets and the messages between the page and its bake worker (`bake-worker.ts`).
 */

export type AssetId = "wdas" | "smoke";

export type WorkerRequest =
  | { type: "load"; asset: AssetId }
  | { type: "bake"; asset: AssetId; resolution: number }
  | { type: "measure"; asset: AssetId; resolution: number };

export interface ErrorStat {
  percent: number;
  /** Voxels compared (every `stride`-th active source voxel). */
  count: number;
  stride: number;
  /** What the error is measured against (a file name). */
  against: string;
}

export type WorkerResponse =
  | {
      type: "loaded";
      asset: AssetId;
      image: Uint32Array;
      gridType: string;
      activeVoxels: number;
      indexDims: [number, number, number];
      fileName: string;
      fileBytes: number;
    }
  | { type: "baked"; asset: AssetId; resolution: number; wvdb: Uint8Array }
  | { type: "sparse-stats"; asset: AssetId; fileGz: number; error: ErrorStat }
  | { type: "dense-stats"; asset: AssetId; resolution: number; error: ErrorStat }
  | { type: "status"; text: string }
  | { type: "error"; message: string };

export const ASSETS: Record<AssetId, { label: string; url: string; reference?: string; stride: number }> = {
  // The Fp8 grid as shipped; measured against the 68 MB .vdb when it is there, else against the Fp8 grid itself.
  wdas: { label: "WDAS cloud (quarter, Fp8)", url: "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb", reference: "/fixtures/wdas/wdas_cloud_quarter.vdb", stride: 31 },
  smoke: { label: "smoke.vdb", url: "/fixtures/vdb-samples/smoke.vdb", stride: 7 },
};
