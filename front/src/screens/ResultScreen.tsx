import { Graphic } from 'components/Graphic';
import React from 'react';
import { useLocation } from 'react-router-dom';
import { IPredictionsFruits } from 'types';

export const ResultScreen: React.FC = () => {
  const location = useLocation();
  const { predictionFruits } = location.state as { predictionFruits: IPredictionsFruits };

  return (
    <>
      <div className="container-fluid">
        <div className="row">
          <div className="col-2"></div>
          <div className="col-8 col-title">
            <h1 className="title">Resultados</h1>
            <p>Os resultados da classificação são exibidos abaixo.</p>
          </div>
          <div className="col-2"></div>
        </div>
        <div className="row">
          <div className="col-md-3"></div>
          <div className="col-6">
            <table className="table table-striped">
              <thead className="table-dark">
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Fruta Obtida</th>
                </tr>
              </thead>
              <tbody>
                {predictionFruits.predictions.map((fruit, index) => (
                  <tr>
                    <th scope="row">{index + 1}</th>
                    <td>{fruit}</td>
                  </tr>
                ))}
              </tbody>

            </table>
          </div>
          <div className="col-3"></div>
        </div>
      </div>
      <Graphic predictions={predictionFruits.predictions} scores={predictionFruits.scores} />
    </>
  );
};

