type Props = {
  error?: string
  children: React.ReactNode
  onSubmit?: (event: React.FormEvent<HTMLFormElement>) => void
}

export function Form({error, children} : Props) {
  return (
    <div className="container-fluid">
      <form>
        {error && 
          <div className="alert alert-danger" role="alert" id="errorMessage">
            {error}
          </div>
        }
        {children}
      </form>
    </div>
  )
}
