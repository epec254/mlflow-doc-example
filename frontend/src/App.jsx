import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Example data from input_data.jsonl (first entry)
const defaultCustomerData = {
  "account": {
    "name": "SkywardAir Transport",
    "industry": "Aviation",
    "size": "Enterprise (4,200 employees)",
    "main_contact": {
      "name": "Captain Olivia Bennett",
      "title": "Director of Operations",
      "email": "o.bennett@skywardair.com"
    },
    "relationship": {
      "customer_since": "2021-08-14",
      "deal_stage": "Mature",
      "account_health": "Good",
      "last_contact_date": "2023-06-11",
      "next_renewal": "2023-08-14"
    }
  },
  "recent_activity": {
    "meetings": [
      {
        "date": "2023-06-11",
        "type": "Renewal Planning",
        "summary": "Discussed flight operations workflow improvements. Olivia wants to expand to maintenance division. Concerned about system availability during peak travel season.",
        "action_items": [
          "Prepare maintenance division implementation plan",
          "Share uptime statistics from past 12 months",
          "Schedule technical review with IT security team"
        ]
      }
    ],
    "product_usage": {
      "active_users": 782,
      "active_users_change": "+4% from last month",
      "most_used_features": [
        "Crew Scheduling",
        "Flight Planning",
        "Compliance Documentation"
      ],
      "least_used_features": [
        "Mobile App",
        "Analytics Dashboard",
        "API Integration"
      ],
      "potential_opportunity": "Mobile app would benefit flight crews accessing schedules remotely"
    },
    "support_tickets": [
      {
        "id": "TK-4622",
        "status": "Resolved Today",
        "issue": "Calendar sync issues with crew schedules",
        "resolution": "Updated timezone handling and recurrence patterns"
      },
      {
        "id": "TK-4630",
        "status": "Open",
        "issue": "Need custom reporting for fuel efficiency metrics",
        "priority": "Medium"
      }
    ]
  },
  "sales_rep": {
    "name": "Jason Mendoza",
    "title": "Aviation Industry Director",
    "signature": "Jason Mendoza\\nAviation Industry Director\\nCloudFlow Inc.\\n(555) 787-3434"
  }
};

function App() {
  const [customerInfo, setCustomerInfo] = useState(JSON.stringify(defaultCustomerData, null, 2));
  const [generatedEmail, setGeneratedEmail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("Checking backend...");
  const [envStatus, setEnvStatus] = useState(null);

  useEffect(() => {
    axios.get('/api/health')
      .then(response => {
        if (response.data.status === 'ok') {
          const clientStatus = response.data.openai_client_initialized ? "Databricks OpenAI client initialized." : "Mock OpenAI client active.";
          setBackendStatus(`Backend is running. ${clientStatus}`);
        } else {
          setBackendStatus("Backend status unknown.");
        }
      })
      .catch(err => {
        console.error("Health check failed:", err);
        setBackendStatus("Backend not reachable. Please start the backend server.");
      });

    // Check backend environment variables
    fetch('http://localhost:8000/api/env-check')
      .then(response => response.json())
      .then(data => {
        setEnvStatus(data);
        console.log('Backend environment variables:', data);
      })
      .catch(error => {
        console.error('Error checking environment variables:', error);
        setEnvStatus({ error: error.message });
      });

    // Log frontend environment variables
    console.log('Frontend environment variables:', {
      VITE_API_URL: import.meta.env.VITE_API_URL,
      VITE_APP_ENV: import.meta.env.VITE_APP_ENV,
    });
  }, []);

  const handleGenerateEmail = async () => {
    setLoading(true);
    setError(null);
    setGeneratedEmail(null);
    try {
      let parsedCustomerInfo;
      try {
        parsedCustomerInfo = JSON.parse(customerInfo);
      } catch (e) {
        setError("Invalid JSON format for customer data. Please check the input.");
        setLoading(false);
        return;
      }

      const response = await axios.post('/api/generate-email/', { 
        customer_info: parsedCustomerInfo
      });
      setGeneratedEmail(response.data);
    } catch (err) {
      console.error("Error generating email:", err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(`Error from backend: ${err.response.data.detail}`);
      } else {
        setError("Failed to generate email. Check the console for more details and ensure the backend is running and configured correctly.");
      }
    }
    setLoading(false);
  };

  return (
    <div className="App">
      {envStatus && (
        <div style={{ 
          padding: '10px', 
          margin: '10px', 
          backgroundColor: envStatus.all_vars_present ? '#e6ffe6' : '#ffe6e6',
          borderRadius: '4px'
        }}>
          <h3>Environment Variables Status</h3>
          <pre>{JSON.stringify(envStatus, null, 2)}</pre>
        </div>
      )}
      <div style={{ 
        padding: '10px', 
        margin: '10px', 
        backgroundColor: '#e6f3ff',
        borderRadius: '4px'
      }}>
        <h3>MLflow Tracing</h3>
        <button 
          onClick={() => window.open(`${import.meta.env.VITE_DATABRICKS_HOST}/ml/experiments/${import.meta.env.VITE_MLFLOW_EXPERIMENT_ID}/traces`, '_blank')}
          style={{
            padding: '8px 16px',
            backgroundColor: '#0066cc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          View MLflow Tracing
        </button>
      </div>
      <header className="App-header">
        <h1>Personalized Email Generator</h1>
        <p className="backend-status">{backendStatus}</p>
      </header>
      <main>
        <div className="input-section">
          <h2>Customer Data (JSON)</h2>
          <p>Paste your customer JSON data below. An example is pre-filled.</p>
          <textarea
            value={customerInfo}
            onChange={(e) => setCustomerInfo(e.target.value)}
            rows={25}
            cols={80}
            placeholder='Paste customer JSON data here'
          />
          <button onClick={handleGenerateEmail} disabled={loading}>
            {loading ? 'Generating...' : 'Generate Email'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <pre>{error}</pre>
          </div>
        )}

        {generatedEmail && (
          <div className="output-section">
            <h2>Generated Email</h2>
            <div className="email-output">
              <h3>Subject: {generatedEmail.subject_line}</h3>
              <pre className="email-body">{generatedEmail.body}</pre>
            </div>
          </div>
        )}
      </main>
      <footer>
        <p>CloudFlow Inc. Demo</p>
      </footer>
    </div>
  );
}

export default App; 