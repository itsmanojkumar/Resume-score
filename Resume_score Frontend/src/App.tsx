import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import ResumeScoringSoftware from './resume_score';
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <ResumeScoringSoftware></ResumeScoringSoftware>
      </div>
    </>
  ) 
}

export default App
