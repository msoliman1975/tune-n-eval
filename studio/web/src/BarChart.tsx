import * as echarts from "echarts";
import { useEffect, useRef } from "react";

export type BarSeries = { name: string; values: (number | null)[]; color: string };

/** Grouped bars: one group per category, one bar per series (model), hover tooltip per bar.
 * Thin bars with 4px rounded data ends, a 2px gap between neighbours, recessive grid, legend for 2+ series. */
export function BarChart({ categories, series, horizontal = false, format = (v) => String(v), max, height = 260, integer = false }: {
  categories: string[]; series: BarSeries[]; horizontal?: boolean; format?: (v: number) => string; max?: number; height?: number;
  integer?: boolean;  // counts: whole-number ticks only
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!el.current) return;
    chart.current = echarts.init(el.current);
    const ro = new ResizeObserver(() => chart.current?.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); chart.current?.dispose(); };
  }, []);

  useEffect(() => {
    const css = getComputedStyle(document.documentElement);
    const v = (n: string) => css.getPropertyValue(n).trim();
    const [muted, grid, fg, surface] = [v("--color-muted"), v("--color-grid"), v("--color-fg"), v("--color-surface")];
    const value = { type: "value", max, minInterval: integer ? 1 : undefined, axisLabel: { color: muted, formatter: (x: number) => format(x) }, splitLine: { lineStyle: { color: grid } } };
    const cat = { type: "category", data: categories, axisLabel: { color: muted, interval: categories.length > 8 ? "auto" : 0, hideOverlap: true }, axisTick: { show: false }, axisLine: { lineStyle: { color: grid } } };
    const radius = horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0];
    chart.current?.setOption({
      animation: false,
      grid: { left: horizontal ? 8 : 48, right: 16, top: series.length > 1 ? 32 : 12, bottom: 8, containLabel: true },
      legend: series.length > 1 ? { top: 0, left: 0, itemWidth: 10, itemHeight: 10, textStyle: { color: fg } } : undefined,
      tooltip: {
        trigger: "item", backgroundColor: surface, borderColor: grid, textStyle: { color: fg },
        formatter: (p: { seriesName: string; name: string; value: number | null }) => `${p.name}<br/>${p.seriesName}: <b>${p.value === null ? "—" : format(p.value)}</b>`,
      },
      xAxis: horizontal ? value : cat,
      yAxis: horizontal ? { ...cat, inverse: true } : value,
      series: series.map((s) => ({
        name: s.name, type: "bar", data: s.values, barMaxWidth: 18, barGap: "12%", barCategoryGap: "30%",
        itemStyle: { color: s.color, borderRadius: radius, borderColor: surface, borderWidth: 1 },
        emphasis: { focus: "none", itemStyle: { opacity: 0.85 } },
      })),
    }, true);
  }, [categories, series, horizontal, max, integer]);

  return <div ref={el} style={{ height }} className="w-full" role="img" aria-label={`Bar chart: ${series.map((s) => s.name).join(", ")} by ${categories.join(", ")}`} />;
}
