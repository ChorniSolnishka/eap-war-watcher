import React, { useCallback, useMemo, useRef, useState } from "react";
import {
  createWarAndUploadScreenshots,
  processAndGetPreviews,
  downloadWarReport,
} from "./lib/api";
import ImageZoom from "./components/ImageZoom";

/** Small helper to join class names. */
function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

function LabeledImage({
  title,
  url,
  onClick,
}: {
  title: string;
  url?: string | null;
  onClick?: () => void;
}) {
  const src = url || null;
  return (
    <figure className="rounded-2xl overflow-hidden bg-slate-900/30 border border-white/5 backdrop-blur">
      <figcaption className="px-3 pt-2 pb-1 text-xs text-slate-200/80">{title}</figcaption>
      {src ? (
        <button
          type="button"
          onClick={onClick}
          className="block w-full h-64 bg-black/40 hover:bg-black/30 transition"
          title="Click to zoom"
        >
          <img src={src} alt={title} className="w-full h-full object-contain" />
        </button>
      ) : (
        <div className="h-64 flex items-center justify-center text-xs text-slate-300/70">
          No preview
        </div>
      )}
    </figure>
  );
}

export default function App() {
  // Alliances
  const [myAlliance, setMyAlliance] = useState("");
  const [enemyAlliance, setEnemyAlliance] = useState("");

  // Files to upload
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Backend state
  const warIdRef = useRef<number | null>(null);
  const [lastScreenId, setLastScreenId] = useState<number | null>(null);

  // Previews from API (absolute URLs)
  const [prevDebugUrl, setPrevDebugUrl] = useState<string | null>(null);
  const [latestDebugUrl, setLatestDebugUrl] = useState<string | null>(null);

  // Skip rows text
  const [skipRowsText, setSkipRowsText] = useState("");

  // Flags
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // Lightbox / zoom state
  const [zoomSrc, setZoomSrc] = useState<string | null>(null);
  const [zoomAlt, setZoomAlt] = useState<string>("");

  const canUpload = useMemo(
    () =>
      myAlliance.trim().length > 0 &&
      enemyAlliance.trim().length > 0 &&
      files.length > 0,
    [myAlliance, enemyAlliance, files]
  );
  const canProcess = useMemo(
    () => !!warIdRef.current && Number.isFinite(lastScreenId),
    [lastScreenId]
  );

  const onPickFiles = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files ? Array.from(e.target.files) : [];
    setFiles(f);
  }, []);

  const handleUpload = async () => {
    if (!canUpload) return;
    try {
      setIsUploading(true);
      setPrevDebugUrl(null);
      setLatestDebugUrl(null);
      setLastScreenId(null);

      const { war, previews } = await createWarAndUploadScreenshots({
        myAlliance,
        enemyAlliance,
        files,
      });

      warIdRef.current = war?.war_id ?? null;

      // Latest & previous debug.png (absolute URLs from backend)
      setLatestDebugUrl(previews?.latest?.debug_url ?? null);
      setPrevDebugUrl(previews?.previous?.debug_url ?? null);
      setLastScreenId(previews?.latest?.screen_id ?? null);
    } catch (err: any) {
      console.error(err);
      alert(err?.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcessAndExport = async () => {
    if (!canProcess || warIdRef.current == null || lastScreenId == null) return;
    try {
      setIsProcessing(true);

      const skipRows = skipRowsText
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean)
        .map((x) => Number(x))
        .filter((n) => Number.isFinite(n) && n > 0);

      const res = await processAndGetPreviews({
        warId: warIdRef.current,
        lastScreenId,
        skipRows,
      });

      // Refresh preview URLs (server returns absolute, possibly updated)
      setLatestDebugUrl(res?.previews?.latest?.debug_url ?? latestDebugUrl);
      setPrevDebugUrl(res?.previews?.previous?.debug_url ?? prevDebugUrl);

      await downloadWarReport(warIdRef.current);
    } catch (err: any) {
      console.error(err);
      alert(err?.message || "Processing failed");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="relative min-h-screen text-slate-100">
      {/* Background image (served from /public/background.png) */}
      <div
          className="fixed inset-0 -z-10 bg-cover bg-center bg-no-repeat opacity-80"
          style={{backgroundImage: `url(/background.png)`}}
          aria-hidden
      />
      {/* A subtle gradient overlay to keep text readable while staying transparent */}
      <div
          className="fixed inset-0 -z-10 bg-gradient-to-br from-slate-950/50 via-slate-950/35 to-slate-900/30"
          aria-hidden
      />
      <div className="relative mx-auto max-w-7xl px-4 py-6">
        <header className="mb-6">
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">War Watcher</h1>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: inputs + single action */}
          <section className="lg:col-span-1">
            {/* More transparent card */}
            {/* Card — lighter background, stronger blur, slightly stronger border for contrast */}
            <div className="rounded-2xl bg-slate-900/30 border border-white/10 shadow-xl backdrop-blur-lg p-5">
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <label className="block text-sm mb-1">My alliance</label>
                  <input
                      className="w-full rounded-lg bg-slate-800/40 border border-white/10 px-3 py-2 outline-none focus:ring-2 ring-blue-400"
                      placeholder="Enter your alliance name"
                      value={myAlliance}
                      onChange={(e) => setMyAlliance(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Enemy alliance</label>
                  <input
                      className="w-full rounded-lg bg-slate-800/40 border border-white/10 px-3 py-2 outline-none focus:ring-2 ring-blue-400"
                      placeholder="Enter enemy alliance name"
                      value={enemyAlliance}
                      onChange={(e) => setEnemyAlliance(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Screenshots</label>
                  <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={onPickFiles}
                      className="block w-full cursor-pointer rounded-lg border border-dashed border-white/20 bg-slate-800/30 p-3 text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-blue-600 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-blue-700"
                  />
                </div>
              </div>

              <div className="mt-4">
                <button
                    onClick={handleUpload}
                    disabled={!canUpload || isUploading}
                    className={cx(
                        "w-full inline-flex items-center justify-center rounded-lg px-4 py-2.5 font-medium transition",
                        "bg-blue-500/90 hover:bg-blue-500 disabled:opacity-50"
                    )}
                >
                  {isUploading ? "Uploading…" : "Upload & Segment"}
                </button>
              </div>
            </div>
          </section>

          {/* Right: previews + process/export */}
          <section className="lg:col-span-2">
            {/* More transparent card */}
            <div className="rounded-2xl bg-slate-900/40 border border-white/5 shadow-xl backdrop-blur-md p-5">
              <h2 className="text-lg font-semibold mb-3">Previews & Export</h2>

              {/* Two previews in one compact row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                <LabeledImage
                  title="Previous (segmented)"
                  url={prevDebugUrl ?? undefined}
                  onClick={() => prevDebugUrl && (setZoomSrc(prevDebugUrl), setZoomAlt("Previous (segmented)"))}
                />
                <LabeledImage
                  title="Latest (segmented)"
                  url={latestDebugUrl ?? undefined}
                  onClick={() => latestDebugUrl && (setZoomSrc(latestDebugUrl), setZoomAlt("Latest (segmented)"))}
                />
              </div>

              {/* Skip rows + process/export */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
                <div className="md:col-span-2">
                  <label className="block text-sm mb-1">Skip rows (comma-separated)</label>
                  <input
                    className="w-full rounded-lg bg-slate-800/40 border border-white/10 px-3 py-2 outline-none focus:ring-2 ring-emerald-400 disabled:opacity-50"
                    placeholder="Example: 1, 3, 5"
                    value={skipRowsText}
                    onChange={(e) => setSkipRowsText(e.target.value)}
                    disabled={!canProcess || isProcessing}
                  />
                  <p className="mt-1 text-[11px] text-slate-300/80">
                    Optional. 1-based indices to ignore on the latest screenshot.
                  </p>
                </div>
                <div>
                  <button
                    onClick={handleProcessAndExport}
                    disabled={!canProcess || isProcessing}
                    className={cx(
                      "w-full inline-flex items-center justify-center rounded-lg px-4 py-2.5 font-medium transition",
                      "bg-emerald-500/90 hover:bg-emerald-500 disabled:opacity-50"
                    )}
                    title={
                      canProcess
                        ? "Process all and export XLSX"
                        : "Upload first to enable processing"
                    }
                  >
                    {isProcessing ? "Processing…" : "Process & Export XLSX"}
                  </button>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Zoom/lightbox */}
        {zoomSrc ? (
          <ImageZoom url={zoomSrc} alt={zoomAlt} onClose={() => setZoomSrc(null)} />
        ) : null}
      </div>
    </div>
  );
}
