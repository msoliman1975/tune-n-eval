import * as echarts from "echarts";
import { useEffect, useRef } from "react";

export type Series = { name: string; data: [number, number][]; color: string; dashed?: boolean };

/** Line chart over training steps. Re-renders in place, so live updates do not flicker. */
export function LineChart({ series, yName, height = 260 }: { series: Series[]; yName: string; height?: number }) {
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
    const muted = css.getPropertyValue("--color-muted").trim();
    const line = css.getPropertyValue("--color-line").trim();
    chart.current?.setOption({
      animation: false,
      grid: { left: 48, right: 16, top: 28, bottom: 32 },
      legend: { top: 0, right: 0, textStyle: { color: muted } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "value", name: "step", nameLocation: "end", axisLabel: { color: muted }, splitLine: { show: false }, axisLine: { lineStyle: { color: line } } },
      yAxis: { type: "value", name: yName, scale: true, axisLabel: { color: muted }, nameTextStyle: { color: muted }, splitLine: { lineStyle: { color: line } } },
      series: series.map((s) => ({
        name: s.name, type: "line", data: s.data, showSymbol: s.data.length < 30, symbolSize: 5,
        lineStyle: { color: s.color, width: 2, type: s.dashed ? "dashed" : "solid" }, itemStyle: { color: s.color },
      })),
    }, true);
  }, [series, yName]);

  return <div ref={el} style={{ height }} className="w-full" />;
}
