import React from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { IPredictionsFruits } from "types";

export function FormScreen() {
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);

    axios
      .post("http://localhost:5000/classify", formData)
      .then((response) => {
        const predictionFruits  : IPredictionsFruits = response.data;

        navigate("/result", { state: { predictionFruits } });
      }).catch((error) => {
        console.error(error);
      });
  };

  return (
    <div className="container">
      <div className="row">
        <div className="col-md-12 col-sm-12 col-lg-12">
          <h3>Envie o seu CSV e descubrar o tipo de frutas</h3>
          <p>Envie um arquivo CSV conforme os campos da tabela abaixo.</p>

          <div className="panel-body row">
            <div className="col-xs-11 table-responsive">
              <table className="table table-striped">
                <thead>
                  <tr>
                    <th scope="col">#</th>
                    <th scope="col">area</th>
                    <th scope="col">perimetro</th>
                    <th scope="col">eixo_maior</th>
                    <th scope="col">eixo_menor</th>
                    <th scope="col">excentricidade</th>
                    <th scope="col">eqdiasq</th>
                    <th scope="col">solidez</th>
                    <th scope="col">area_convexa</th>
                    <th scope="col">extensao</th>
                    <th scope="col">proporcao</th>
                    <th scope="col">redondidade</th>
                    <th scope="col">compactidade</th>
                    <th scope="col">fator_forma_1</th>
                    <th scope="col">fator_forma_2</th>
                    <th scope="col">fator_forma_3</th>
                    <th scope="col">fator_forma_4</th>
                    <th scope="col">RR_media</th>
                    <th scope="col">RG_media</th>
                    <th scope="col">RB_media</th>
                    <th scope="col">RR_dev</th>
                    <th scope="col">RG_dev</th>
                    <th scope="col">RB_dev</th>
                    <th scope="col">RR_inclinacao</th>
                    <th scope="col">RG_inclinacao</th>
                    <th scope="col">RB_inclinacao</th>
                    <th scope="col">RR_curtose</th>
                    <th scope="col">RG_curtose</th>
                    <th scope="col">RB_curtose</th>
                    <th scope="col">RR_entropia</th>
                    <th scope="col">RG_entropia</th>
                    <th scope="col">RB_entropia</th>
                    <th scope="col">RR_all</th>
                    <th scope="col">RG_all</th>
                    <th scope="col">RB_all</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">1</th>
                    <th>422163</th>
                    <th>2378.908</th>
                    <th>837.8484</th>
                    <th>645.6693</th>
                    <th>0.6373</th>
                    <th>733.1539</th>
                    <th>0.9947</th>
                    <th>424428</th>
                    <th>0.7831</th>
                    <th>1.2976</th>
                    <th>0.9374</th>
                    <th>0.875</th>
                    <th>0.002</th>
                    <th>0.0015</th>
                    <th>0.7657</th>
                    <th>0.9936</th>
                    <th>117.4466</th>
                    <th>109.9085</th>
                    <th>95.6774</th>
                    <th>26.5152</th>
                    <th>23.0687</th>
                    <th>30.123</th>
                    <th>-0.5661</th>
                    <th>-0.0114</th>
                    <th>0.6019</th>
                    <th>3.237</th>
                    <th>2.9574</th>
                    <th>4.2287</th>
                    <th>-59191263232,</th>
                    <th>-50714214400</th>
                    <th>-39922372608</th>
                    <th>58.7255</th>
                    <th>54.9554</th>
                    <th>47.84</th>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <hr />

      <div className="row">
        <div className="col">
          <h3>Formulário</h3>
          <form method="POST" onSubmit={handleSubmit}>
            <input type="file" name="file" id="file" />
            <input type="submit" value="Enviar" />
          </form>
        </div>
      </div>
    </div >
  );
}
