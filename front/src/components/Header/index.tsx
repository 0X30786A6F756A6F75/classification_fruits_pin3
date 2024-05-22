import { Link } from "react-router-dom"

export function Header() {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container-fluid">
        <Link className="navbar-brand d-flex align-items-center" to="/">
          <img src="/brand_logo.png" alt="Brand Logo" width="30" height="30" className="me-2" />
          Frutas++
        </Link>

        <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarTogglerDemo01" aria-controls="navbarTogglerDemo01" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse justify-content-end" id="navbarTogglerDemo01">
          <ul className="navbar-nav">
            <li className="nav-item">
              <Link to="/" className="nav-link">Verificar Frutas</Link>
            </li>
            <li className="nav-item">
              <Link to="/graphics" className="nav-link">Graficos</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  )
}

