import { Header } from "./components/Header"
import { Footer } from "./components/Footer"
import "./assets/css/global/styles.css"
import { Route, Routes } from "react-router-dom"
import { HomeScreen } from "screens/HomeScreen"
import { FormScreen } from "screens/FormScreen"
import { ResultScreen } from "screens/ResultScreen"

function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<HomeScreen />} />
        <Route path="/form" element={<FormScreen />} />
        <Route path="/result" element={<ResultScreen />} />
      </Routes>
    </>
  )
}

export default App
