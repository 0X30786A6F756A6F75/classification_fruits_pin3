import { useEffect, useRef } from "react";
import Chart, { ChartData } from "chart.js/auto";
import { IPredictionsFruits } from "types";

type ChartType = 'bar' | 'line' | 'pie';

const backgroundColor = [
  "rgba(255, 99, 132, 0.2)",
  "rgba(54, 162, 235, 0.2)",
  "rgba(255, 206, 86, 0.2)",
  "rgba(75, 192, 192, 0.2)",
  "rgba(153, 102, 255, 0.2)",
  "rgba(255, 159, 64, 0.2)"
];
const borderColor = [
  "rgba(255, 99, 132, 1)",
  "rgba(54, 162, 235, 1)",
  "rgba(255, 206, 86, 1)",
  "rgba(75, 192, 192, 1)",
  "rgba(153, 102, 255, 1)",
  "rgba(255, 159, 64, 1)"
];

export const Graphic: React.FC<IPredictionsFruits> = ({ predictions, scores }) => {
  const barChartRef = useRef<HTMLCanvasElement>(null);
  const distributionChartRef = useRef<HTMLCanvasElement>(null);
  const pieChartRef = useRef<HTMLCanvasElement>(null);

  const chartInstances = useRef<{ [key in ChartType]: Chart | null }>({
    bar: null,
    line: null,
    pie: null
  });

  const destroyChart = (chartType: ChartType) => {
    if (chartInstances.current[chartType]) {
      chartInstances.current[chartType]?.destroy();
      chartInstances.current[chartType] = null;
    }
  };

  const initChart = (ref: React.RefObject<HTMLCanvasElement>, chartType: ChartType, data: ChartData) => {
    const canvas = ref.current;
    if (canvas) {
      destroyChart(chartType);
      chartInstances.current[chartType] = new Chart(canvas, {
        type: chartType,
        data,
        options: {
          scales: {
            y: {
              beginAtZero: chartType !== 'line'
            }
          }
        }
      });
    }
  };

  useEffect(() => {
    const columns = ["BERHI", "DOKOL", "IRAQI", "ROTANA", "SAFAVI", "SOGAY"];

    const data = [
      scores?.BERHI || 0,
      scores?.DOKOL || 0,
      scores?.IRAQI || 0,
      scores?.ROTANA || 0,
      scores?.SAFAVI || 0,
      scores?.SOGAY || 0
    ];

    const barChartData: ChartData = {
      labels: columns,
      datasets: [{
        label: "Frutas",
        data: data,
        backgroundColor,
        borderColor,
        borderWidth: 1,
      }]
    };

    const lineChartData: ChartData = {
      labels: columns,
      datasets: [{
        label: "Frutas",
        data: data,
        fill: false,
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 1,
      }]
    };

    const pieChartData: ChartData = {
      labels: columns,
      datasets: [{
        data: data,
        backgroundColor,
        borderColor,
        borderWidth: 1
      }]
    };

    initChart(barChartRef, 'bar', barChartData);
    initChart(distributionChartRef, 'line', lineChartData);
    initChart(pieChartRef, 'pie', pieChartData);

    // Cleanup charts on component unmount
    return () => {
      destroyChart('bar');
      destroyChart('line');
      destroyChart('pie');
    };
  }, [predictions, scores]);

  return (
    <div className="container-fluid">
      <div className="row justify-content-center">
        <div className="col-sm">
          <h1 className="title">Gráficos</h1>
        </div>
      </div>
      <div className="row justify-content-center">
        <div className="col-6">
          <canvas ref={barChartRef} width="600" height="400"></canvas>
          <hr />
        </div>
      </div>
      <br />
      <div className="row justify-content-center">
        <div className="col-1"></div>
        <div className="col-4">
          <canvas ref={distributionChartRef} width="300" height="300"></canvas>
        </div>
        <div className="col-1"></div>
        <div className="col-4">
          <canvas ref={pieChartRef} width="300" height="300"></canvas>
        </div>
        <div className="col-1"></div>
      </div>
    </div>
  );
};
