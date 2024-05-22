import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

export function Graphic() {
  let pieChartRef = useRef(null);
  let barChartRef = useRef(null);
  let distributionChartRef = useRef(null);

  useEffect(() => {
    const columns = ["Berthi", "Dokol", "IRAQI", "Rotana", "Safavi", "Sogay"];
    const randomData = () => {
      return Array.from({ length: 7 }, () => Math.floor(Math.random() * 100));
    };

    const destroyChart = (chart: Chart) => {
      chart.destroy();
    };

    let backgroundColor = [
      "rgba(255, 99, 132, 0.2)",
      "rgba(54, 162, 235, 0.2)",
      "rgba(255, 206, 86, 0.2)",
      "rgba(75, 192, 192, 0.2)",
      "rgba(153, 102, 255, 0.2)",
      "rgba(255, 159, 64, 0.2)"
    ];
    let borderColor = [
      "rgba(255, 99, 132, 1)",
      "rgba(54, 162, 235, 1)",
      "rgba(255, 206, 86, 1)",
      "rgba(75, 192, 192, 1)",
      "rgba(153, 102, 255, 1)",
      "rgba(255, 159, 64, 1)"
    ];

    let barChartRef = new Chart("bar", {
      type: "bar",
      data: {
        labels: columns,
        datasets: [{
          label: "Frutas",
          data: randomData(),
          backgroundColor: backgroundColor,
          borderWidth: 1,
          borderColor: borderColor,
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });

    distributionChartRef = new Chart("line", distributionChartRef.current, {
      type: "line",
      data: {
        labels: columns,
        datasets: [{
          label: "Frutas",
          data: randomData(),
          fill: false,
          borderColor: "rgba(255, 99, 132, 1)",
          borderWidth: 1,
          borderColor: borderColor,
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: false
          }
        }
      }
    });

    pieChartRef = new Chart("pie", {
      type: "pie",
      data: {
        labels: columns,
        datasets: [{
          data: [12, 19, 3, 5, 2, 3],
          backgroundColor: backgroundColor,
          borderColor: borderColor,
          borderWidth: 1
        }]
      }
    });
  }, []);

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
}

