import axios from "axios";
import { Form } from "components/Form"
import { Graphic } from "components/Graphic"
import { useEffect, useState } from "react";
import { IPredictionsFruits } from "types";

export function HomeScreen() {
  const [fruist, setFruits] = useState<IPredictionsFruits>({
    'predictions': [],
    'scores': { "BERHI": 0, "DOKOL": 0, "IRAQI": 0, "ROTANA": 0, "SAFAVI": 0, "SOGAY": 0 }
  });
  console.log(fruist);

  useEffect(() => {
    axios
      .get("http://localhost:5000/modelScores")
      .then((response) => {
        setFruits(response.data);
      }).catch((error) => {
        // console.error(error);
      });
  }, []);


  return (
      <Graphic predictions={fruist.predictions} scores={fruist.scores} />
  )
}
