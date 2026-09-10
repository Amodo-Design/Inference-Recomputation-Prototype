// PNG export for chart chrome: clone the chart's <svg>, inline the card's
// real background + font-family, rasterize at 2x with the HTML legend
// redrawn beneath the panels, and trigger a browser download.
// Canvas/image-loading is DOM-only so this has no unit coverage; visual
// smoke checks cover it.

const EXPORT_BACKGROUND = "#fcfbf4"; // fallback when no .chart-card ancestor resolves
const EXPORT_SCALE = 2;
const LEGEND_FONT_SIZE = 12;
const LEGEND_ROW_HEIGHT = 22;
const LEGEND_SWATCH = 12;
const LEGEND_PADDING = 12;
const LEGEND_ITEM_GAP = 20;

type ExportLegendItem = {
  label: string;
  color: string;
  dashed: boolean;
  opacity: number;
};

// The legend is HTML (a sibling of the exported .chart-area), so it never
// survives SVG serialization — collect its items from the DOM and redraw
// them onto the canvas instead. Hidden (toggled-off) series keep their
// dimmed opacity so the export matches what the user sees.
function collectLegend(container: HTMLDivElement): ExportLegendItem[] {
  const frame = container.closest(".chart-frame");
  if (!frame) return [];
  return Array.from(frame.querySelectorAll(".chart-legend-item")).map((el) => {
    const swatch = el.querySelector(".chart-legend-swatch");
    const dashed = swatch?.classList.contains("chart-legend-swatch-dashed") ?? false;
    const swatchStyle = swatch ? getComputedStyle(swatch) : null;
    const color =
      (dashed ? swatchStyle?.borderTopColor : swatchStyle?.backgroundColor) || "#000";
    return {
      label: el.querySelector(".chart-legend-label")?.textContent?.trim() ?? "",
      color,
      dashed,
      opacity: Number(getComputedStyle(el as Element).opacity || "1"),
    };
  });
}

function cardBackground(container: HTMLDivElement): string {
  const card = container.closest(".chart-card");
  if (!card) return EXPORT_BACKGROUND;
  const bg = getComputedStyle(card).backgroundColor;
  // Transparent computed backgrounds (rgba(...,0)) fall back to the default.
  if (!bg || bg === "transparent" || /rgba\(.*,\s*0\)$/.test(bg)) return EXPORT_BACKGROUND;
  return bg;
}

// Greedy row wrap of legend items into totalWidth; returns per-item
// positions plus the block height so the canvas can be sized before drawing.
function layoutLegend(
  ctx: CanvasRenderingContext2D,
  items: ExportLegendItem[],
  totalWidth: number,
  fontFamily: string
): { positions: { x: number; y: number }[]; height: number } {
  if (items.length === 0) return { positions: [], height: 0 };
  ctx.font = `${LEGEND_FONT_SIZE}px ${fontFamily}`;
  const positions: { x: number; y: number }[] = [];
  let x = LEGEND_PADDING;
  let row = 0;
  for (const item of items) {
    const itemWidth = LEGEND_SWATCH + 6 + ctx.measureText(item.label).width;
    if (x + itemWidth > totalWidth - LEGEND_PADDING && x > LEGEND_PADDING) {
      row += 1;
      x = LEGEND_PADDING;
    }
    positions.push({ x, y: row * LEGEND_ROW_HEIGHT });
    x += itemWidth + LEGEND_ITEM_GAP;
  }
  return { positions, height: (row + 1) * LEGEND_ROW_HEIGHT + LEGEND_PADDING };
}

function drawLegend(
  ctx: CanvasRenderingContext2D,
  items: ExportLegendItem[],
  positions: { x: number; y: number }[],
  yOffset: number,
  fontFamily: string
): void {
  ctx.font = `${LEGEND_FONT_SIZE}px ${fontFamily}`;
  ctx.textBaseline = "middle";
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const { x, y } = positions[i];
    const cy = yOffset + y + LEGEND_ROW_HEIGHT / 2;
    ctx.globalAlpha = item.opacity;
    if (item.dashed) {
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x, cy);
      ctx.lineTo(x + LEGEND_SWATCH, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.fillStyle = item.color;
      ctx.fillRect(x, cy - LEGEND_SWATCH / 2, LEGEND_SWATCH, LEGEND_SWATCH);
    }
    ctx.fillStyle = "#3a3a33";
    ctx.fillText(item.label, x + LEGEND_SWATCH + 6, cy);
  }
  ctx.globalAlpha = 1;
}

export async function exportPng(
  container: HTMLDivElement,
  slug: string
): Promise<void> {
  const svgs = Array.from(container.querySelectorAll("svg"));
  if (svgs.length === 0) return;

  const fontFamily =
    getComputedStyle(container).fontFamily ||
    '"IBM Plex Sans", "Helvetica Neue", sans-serif';

  // Clone + inline computed styles for every panel svg, and measure each
  // panel independently so multi-panel charts (verification-margin,
  // pass-fail-by-output-tokens, the faceted heatmap) export as one image
  // with every panel stacked vertically instead of just the first svg.
  const panels = svgs.map((svg) => {
    const rect = svg.getBoundingClientRect();
    const width = Math.max(
      1,
      Math.round(rect.width || Number(svg.getAttribute("width")) || 800)
    );
    const height = Math.max(
      1,
      Math.round(rect.height || Number(svg.getAttribute("height")) || 400)
    );

    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("font-family", fontFamily);
    clone.style.fontFamily = fontFamily;

    inlineComputedStyles(svg, clone);

    const background = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "rect"
    );
    background.setAttribute("x", "0");
    background.setAttribute("y", "0");
    background.setAttribute("width", String(width));
    background.setAttribute("height", String(height));
    background.setAttribute("fill", cardBackground(container));
    clone.insertBefore(background, clone.firstChild);

    const svgString = new XMLSerializer().serializeToString(clone);
    const svgBlob = new Blob([svgString], {
      type: "image/svg+xml;charset=utf-8",
    });
    return { width, height, url: URL.createObjectURL(svgBlob) };
  });

  const totalWidth = Math.max(...panels.map((p) => p.width));
  const panelsHeight = panels.reduce((sum, p) => sum + p.height, 0);
  const background = cardBackground(container);
  const legendItems = collectLegend(container);

  try {
    const images = await Promise.all(panels.map((p) => loadImage(p.url)));
    const canvas = document.createElement("canvas");
    // Provisional context purely for measuring legend text before sizing.
    const measureCtx = document.createElement("canvas").getContext("2d");
    const legendLayout = measureCtx
      ? layoutLegend(measureCtx, legendItems, totalWidth, fontFamily)
      : { positions: [], height: 0 };
    canvas.width = totalWidth * EXPORT_SCALE;
    canvas.height = (panelsHeight + legendLayout.height) * EXPORT_SCALE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(EXPORT_SCALE, EXPORT_SCALE);

    let y = 0;
    for (let i = 0; i < panels.length; i++) {
      ctx.drawImage(images[i], 0, y, panels[i].width, panels[i].height);
      y += panels[i].height;
    }
    if (legendLayout.positions.length > 0) {
      drawLegend(ctx, legendItems, legendLayout.positions, y, fontFamily);
    }

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((b) => resolve(b), "image/png");
    });
    if (blob) downloadBlob(blob, filenameFor(slug));
  } finally {
    panels.forEach((p) => URL.revokeObjectURL(p.url));
  }
}

// Computed style properties to inline onto the clone so CSS-class/var-styled
// marks and axis text survive serialization outside the DOM (the clone has
// no stylesheet access once detached into a standalone svg document).
const STYLE_PROPS = [
  "fill",
  "stroke",
  "stroke-width",
  "stroke-dasharray",
  "opacity",
  "fill-opacity",
  "stroke-opacity",
  "font-family",
  "font-size",
  "font-weight",
  "color",
] as const;

// Walk the original and clone trees in lockstep (querySelectorAll("*") plus
// the roots gives identical order on both, since the clone is a deep clone
// of the original) so getComputedStyle on the original — which resolves CSS
// variables and class-based rules — can be copied onto the clone as inline
// styles.
function inlineComputedStyles(original: SVGSVGElement, clone: SVGSVGElement): void {
  const originalNodes = [original, ...Array.from(original.querySelectorAll("*"))];
  const cloneNodes = [clone, ...Array.from(clone.querySelectorAll("*"))];
  if (originalNodes.length !== cloneNodes.length) return;

  for (let i = 0; i < originalNodes.length; i++) {
    const originalEl = originalNodes[i];
    const cloneEl = cloneNodes[i];
    if (!(cloneEl instanceof SVGElement)) continue;
    const computed = getComputedStyle(originalEl);
    for (const prop of STYLE_PROPS) {
      const value = computed.getPropertyValue(prop);
      if (value) cloneEl.style.setProperty(prop, value);
    }
  }
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to rasterize chart svg"));
    img.src = src;
  });
}

function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Defer revocation so a slow/late-triggered download isn't cancelled by an
  // immediately-revoked object URL (race-cancel hardening).
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function filenameFor(slug: string): string {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const yyyymmdd = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}`;
  const hhmm = `${pad(now.getHours())}${pad(now.getMinutes())}`;
  return `infver-${slug}-${yyyymmdd}-${hhmm}.png`;
}
