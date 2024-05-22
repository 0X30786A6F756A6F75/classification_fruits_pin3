import { useEffect, useRef } from "react";
import Chart, { ChartData } from "chart.js/auto";

type ChartType = 'bar' | 'line' | 'pie';

interface GraphicProps { }

const randomData = (length: number) => Array.from({ length }, () => Math.floor(Math.random() * 100));
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

export const Graphic: React.FC<GraphicProps> = () => {
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
    const columns = ["Berthi", "Dokol", "IRAQI", "Rotana", "Safavi", "Sogay"];

    const barChartData: ChartData = {
      labels: columns,
      datasets: [{
        label: "Frutas",
        data: randomData(columns.length),
        backgroundColor,
        borderColor,
        borderWidth: 1,
      }]
    };

    const lineChartData: ChartData = {
      labels: columns,
      datasets: [{
        label: "Frutas",
        data: randomData(columns.length),
        fill: false,
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 1,
      }]
    };

    const pieChartData: ChartData = {
      labels: columns,
      datasets: [{
        data: [12, 19, 3, 5, 2, 3],
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
  }, [initChart]);

  return (
    <div className="container-fluid" id="content-main">
      <div className="row justify-content-center">
        <div className="col-12 text-center mb-4 mx-2">
          <h1 className="h1">Gráficos</h1>
        </div>
        <div className="col-12 col-lg-6 mb-4 mb-lg-2 mx-2">
          <canvas ref={barChartRef} width="600" height="400"></canvas>
          <hr />
        </div>
      </div>
      <div className="row justify-content-center mt-4">
        <div className="col-8 col-lg-4 col-sm-10 mx-2 mb-4">
          <canvas ref={distributionChartRef} width="300" height="300"></canvas>
        </div>
        <div className="col-1"></div>
        <div className="col-8 col-lg-4 col-sm-10 mx-2 mb-4">
          <canvas ref={pieChartRef} width="300" height="300"></canvas>
        </div>
      </div>
    </div>
		);
};
