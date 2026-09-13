import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './app/App'
import './styles/index.css'
import './styles/integrations/experiment-home.css'
import './styles/integrations/experiment-navigation.css'
import './styles/integrations/experiment-settings.css'
import './styles/integrations/mas-debug.css'
import './styles/integrations/resources.css'
import './styles/integrations/model-catalog.css'
import './styles/integrations/model-binding.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
