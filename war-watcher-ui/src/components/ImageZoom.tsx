import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * ImageZoom — lightweight zoom/pan modal with no dependencies.
 *
 * Features:
 *  - Click to open (you control from parent)
 *  - ESC to close, click backdrop to close, Close button
 *  - Double-click toggles 1x <-> 2x at pointer position
 *  - Mouse wheel zoom (centered around cursor)
 *  - Drag to pan (Pointer Events; works on touch too)
 *
 * Accessibility:
 *  - role="dialog" + aria-modal
 *  - Focus trap (initial focus on container)
 */
export default function ImageZoom({
  url,
  alt = "",
  onClose,
}: {
  url: string;
  alt?: string;
  onClose: () => void;
}) {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Transform state
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0); // translateX
  const [ty, setTy] = useState(0); // translateY

  // Dragging state
  const dragging = useRef(false);
  const dragStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  // Disable body scroll while the dialog is open
  useEffect(() => {
    const { overflow } = document.body.style;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = overflow;
    };
  }, []);

  // Escape to close + initial focus
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      // small keyboard zoom helpers
      if (e.key === "+" || e.key === "=") zoomBy(0.25);
      if (e.key === "-" || e.key === "_") zoomBy(-0.25);
      if (e.key === "0") resetTransform();
    };
    window.addEventListener("keydown", onKey);
    overlayRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Helpers
  const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));
  const zoomBy = (delta: number, cx?: number, cy?: number) => {
    setScale((prev) => {
      const next = clamp(prev + delta, 0.5, 4);
      // Re-center around cursor (cx, cy) in container coords
      if (containerRef.current && cx != null && cy != null) {
        const k = next / prev;
        setTx((t) => cx - k * (cx - t));
        setTy((t) => cy - k * (cy - t));
      }
      return next;
    });
  };
  const resetTransform = () => {
    setScale(1);
    setTx(0);
    setTy(0);
  };

  // Wheel zoom around pointer
  const onWheel: React.WheelEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault(); // important to avoid page scroll under the overlay
    const rect = containerRef.current?.getBoundingClientRect();
    const cx = rect ? e.clientX - rect.left : 0;
    const cy = rect ? e.clientY - rect.top : 0;
    const delta = e.deltaY > 0 ? -0.2 : 0.2; // trackpads included
    zoomBy(delta, cx, cy);
  };

  // Pointer drag handlers
  const onPointerDown: React.PointerEventHandler<HTMLDivElement> = (e) => {
    dragging.current = true;
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
    dragStart.current = { x: e.clientX, y: e.clientY, tx, ty };
  };
  const onPointerMove: React.PointerEventHandler<HTMLDivElement> = (e) => {
    if (!dragging.current || !dragStart.current) return;
    const { x, y, tx: sx, ty: sy } = dragStart.current;
    setTx(sx + (e.clientX - x));
    setTy(sy + (e.clientY - y));
  };
  const onPointerUp: React.PointerEventHandler<HTMLDivElement> = (e) => {
    dragging.current = false;
    dragStart.current = null;
    (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
  };

  // Double click toggle 1x <-> 2x at pointer
  const onDoubleClick: React.MouseEventHandler<HTMLDivElement> = (e) => {
    const rect = containerRef.current?.getBoundingClientRect();
    const cx = rect ? e.clientX - rect.left : 0;
    const cy = rect ? e.clientY - rect.top : 0;
    if (scale <= 1) {
      setScale(2);
      // center at pointer
      setTx((t) => cx - 2 * (cx - t));
      setTy((t) => cy - 2 * (cy - t));
    } else {
      resetTransform();
    }
  };

  // Compose transform style
  const transform = useMemo(() => `translate(${tx}px, ${ty}px) scale(${scale})`, [tx, ty, scale]);

  // Portal to body
  return createPortal(
    <div
      ref={overlayRef}
      role="dialog"
      aria-modal="true"
      tabIndex={-1}
      className="fixed inset-0 z-50 flex flex-col bg-black/70"
      onClick={(e) => {
        // click outside (backdrop) closes
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* Controls */}
      <div className="flex items-center gap-2 p-3 text-sm">
        <button
          className="rounded-md bg-white/10 hover:bg-white/20 px-3 py-1"
          onClick={() => zoomBy(-0.25)}
          title="Zoom out (-)"
        >
          –
        </button>
        <button
          className="rounded-md bg-white/10 hover:bg-white/20 px-3 py-1"
          onClick={() => zoomBy(0.25)}
          title="Zoom in (+)"
        >
          +
        </button>
        <button
          className="rounded-md bg-white/10 hover:bg-white/20 px-3 py-1"
          onClick={resetTransform}
          title="Reset (0)"
        >
          Reset
        </button>
        <div className="ml-auto" />
        <button
          className="rounded-md bg-white/10 hover:bg-white/20 px-3 py-1"
          onClick={onClose}
          title="Close (Esc)"
        >
          Close
        </button>
      </div>

      {/* Image stage */}
      <div
        ref={containerRef}
        className="relative flex-1 overflow-hidden touch-none select-none"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={onDoubleClick}
        // Ensures we get pointer events and disable native touch actions for panning
        style={{ touchAction: "none", cursor: dragging.current ? "grabbing" : "grab" }}
      >
        <img
          src={url}
          alt={alt}
          className="absolute top-1/2 left-1/2 max-h-none max-w-none"
          style={{
            transform,
            transformOrigin: "0 0",
            // Center the image initially (tx/ty start at 0 => center via translate(-50%,-50%))
            translate: "-50% -50%",
          }}
          draggable={false}
        />
      </div>
    </div>,
    document.body
  );
}
