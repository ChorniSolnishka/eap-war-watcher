// src/lib/api.ts
// Simple API helpers and types for War Watcher frontend.

export const API_ORIGIN = "http://localhost:8000";
export const API_BASE = `${API_ORIGIN}/api`;

export type UploadResponse = {
  war: { war_id: number; my_alliance_id: number; enemy_alliance_id: number };
  uploads: any;
  previews: {
    latest?: { screen_id: number; rows_detected: number; debug_url: string | null } | null;
    previous?: { screen_id: number; rows_detected: number; debug_url: string | null } | null;
  };
};

export type ProcessResponse = {
  war_id: number;
  previews: {
    latest?: { screen_id: number; rows_detected: number; debug_url: string | null } | null;
    previous?: { screen_id: number; rows_detected: number; debug_url: string | null } | null;
  };
  processing: any;
};

export async function createWarAndUploadScreenshots(params: {
  myAlliance: string;
  enemyAlliance: string;
  files: File[];
}): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("my_alliance", params.myAlliance);
  fd.append("enemy_alliance", params.enemyAlliance);
  for (const f of params.files) fd.append("files", f);

  const res = await fetch(`${API_BASE}/uploads/war/screenshots`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as UploadResponse;
}

export async function processAndGetPreviews(params: {
  warId: number;
  lastScreenId: number;
  skipRows: number[];
}): Promise<ProcessResponse> {
  const res = await fetch(
    `${API_BASE}/uploads/${params.warId}/screenshots/${params.lastScreenId}/process`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skip_rows: params.skipRows }),
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as ProcessResponse;
}

export async function downloadWarReport(warId: number): Promise<void> {
  const url = `${API_BASE}/export/wars/${warId}/xlsx`;
  // robust cross-browser open
  const w = window.open(url, "_blank", "noopener,noreferrer");
  if (!w) {
    // fallback: direct location change
    window.location.href = url;
  }
}
