import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [customerData, setCustomerData] = useState(null);
  const [userInstructions, setUserInstructions] = useState('');
  const [generatedEmail, setGeneratedEmail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingCompanies, setLoadingCompanies] = useState(true);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("Checking backend...");
  const [envStatus, setEnvStatus] = useState(null);
  // Feedback state
  const [feedbackRating, setFeedbackRating] = useState(null); // 'up' or 'down'
  const [feedbackComment, setFeedbackComment] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  useEffect(() => {
    // Check backend health
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

    // Load companies
    loadCompanies();
  }, []);

  const loadCompanies = async () => {
    try {
      setLoadingCompanies(true);
      const response = await axios.get('/api/companies');
      setCompanies(response.data);
    } catch (err) {
      console.error("Error loading companies:", err);
      setError("Failed to load companies");
    } finally {
      setLoadingCompanies(false);
    }
  };

  const handleCompanySelect = async (companyName) => {
    setSelectedCompany(companyName);
    setSearchQuery(companyName);
    setShowDropdown(false);
    setUserInstructions(''); // Reset instructions when changing companies
    
    if (!companyName) {
      setCustomerData(null);
      return;
    }

    try {
      const response = await axios.get(`/api/customer/${encodeURIComponent(companyName)}`);
      setCustomerData(response.data);
      setError(null);
    } catch (err) {
      console.error("Error loading customer data:", err);
      setError("Failed to load customer data");
      setCustomerData(null);
    }
  };

  const handleSearchChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);
    setShowDropdown(true);
    
    // If search is cleared, also clear selection
    if (!value) {
      setSelectedCompany('');
      setCustomerData(null);
    }
  };

  const filteredCompanies = companies.filter(company =>
    company.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const updateNestedField = (path, value) => {
    setCustomerData(prev => {
      const newData = JSON.parse(JSON.stringify(prev)); // Deep copy
      const keys = path.split('.');
      let current = newData;
      
      for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
      }
      
      current[keys[keys.length - 1]] = value;
      return newData;
    });
  };

  const handleGenerateEmail = async () => {
    if (!customerData) {
      setError("Please select a company first");
      return;
    }

    setLoading(true);
    setError(null);
    setGeneratedEmail(null);
    // Reset feedback when generating new email
    setFeedbackRating(null);
    setFeedbackComment('');
    setFeedbackSubmitted(false);
    
    try {
      // Add user instructions to the customer data
      const requestData = {
        ...customerData,
        user_instructions_for_email: userInstructions
      };

      const response = await axios.post('/api/generate-email/', { 
        customer_info: requestData
      });
      setGeneratedEmail(response.data);
    } catch (err) {
      console.error("Error generating email:", err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(`Error from backend: ${err.response.data.detail}`);
      } else {
        setError("Failed to generate email. Check the console for more details.");
      }
    }
    setLoading(false);
  };

  const handleFeedbackRating = (rating) => {
    setFeedbackRating(rating);
    setFeedbackSubmitted(false);
  };

  const handleFeedbackSubmit = () => {
    if (feedbackRating) {
      // Here you could send feedback to backend
      console.log('Feedback submitted:', {
        rating: feedbackRating,
        comment: feedbackComment,
        email: generatedEmail
      });
      setFeedbackSubmitted(true);
      
      // You could add an API call here to save feedback
      // axios.post('/api/feedback', { rating: feedbackRating, comment: feedbackComment, email_id: generatedEmail.id })
    }
  };

  return (
    <div className="App">
      {envStatus && envStatus.all_vars_present && (
        <div className="mlflow-banner">
          <h3>MLflow Tracing</h3>
          <button 
            className="mlflow-button"
            onClick={() => window.open(`${import.meta.env.VITE_DATABRICKS_HOST}/ml/experiments/${import.meta.env.VITE_MLFLOW_EXPERIMENT_ID}/traces`, '_blank')}
          >
            View MLflow Tracing
          </button>
        </div>
      )}
      
      <header className="App-header">
        <h1>Personalized Email Generator</h1>
        <p className="backend-status">{backendStatus}</p>
      </header>
      
      <main>
        <div className="panels-container">
          {/* Left Panel - Input */}
          <div className="panel panel-left">
            <div className="form-section">
              <div className="section-header">
                <h2>Customer Information</h2>
              </div>
              
              {/* Company Selector */}
              <div className="form-group">
                <label htmlFor="company-select">Select Company</label>
                <div className="typeahead-container">
                  <input
                    id="company-select"
                    type="text"
                    value={searchQuery}
                    onChange={handleSearchChange}
                    onFocus={() => setShowDropdown(true)}
                    onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                    className="company-select"
                    placeholder="Type to search companies..."
                    disabled={loadingCompanies}
                  />
                  {showDropdown && filteredCompanies.length > 0 && (
                    <div className="dropdown">
                      {filteredCompanies.map((company) => (
                        <div
                          key={company.name}
                          className={`dropdown-item ${selectedCompany === company.name ? 'selected' : ''}`}
                          onMouseDown={() => handleCompanySelect(company.name)}
                        >
                          {company.name}
                        </div>
                      ))}
                    </div>
                  )}
                  {showDropdown && searchQuery && filteredCompanies.length === 0 && (
                    <div className="dropdown">
                      <div className="dropdown-item no-results">
                        No companies found matching "{searchQuery}"
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* User Instructions - Moved here */}
              {customerData && (
                <div className="instructions-section">
                  <div className="form-section-header">
                    <h3>Email Instructions</h3>
                  </div>
                  <div className="form-group">
                    <label htmlFor="user-instructions">
                      Add any specific instructions or context for the email
                    </label>
                    <textarea
                      id="user-instructions"
                      value={userInstructions}
                      onChange={(e) => setUserInstructions(e.target.value)}
                      placeholder="E.g., Mention the upcoming product launch, emphasize our new pricing model, schedule a demo for next week..."
                      rows={4}
                      className="instructions-textarea"
                    />
                  </div>
                </div>
              )}

              {/* Generate Email Button */}
              {customerData && (
                <div className="generate-button-section">
                  <button 
                    onClick={handleGenerateEmail} 
                    disabled={loading}
                    className="generate-button"
                  >
                    {loading ? 'Generating 🚀' : 'Generate Email 👉'}
                  </button>
                </div>
              )}

              {/* Customer Data Form */}
              {customerData && (
                <>
                  <div className="customer-form">
                    {/* Account Information */}
                    <div className="form-section-header">
                      <h3>Account Details</h3>
                    </div>
                    
                    <div className="form-row">
                      <div className="form-group">
                        <label>Industry</label>
                        <input
                          type="text"
                          value={customerData.account.industry}
                          onChange={(e) => updateNestedField('account.industry', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Size</label>
                        <input
                          type="text"
                          value={customerData.account.size}
                          onChange={(e) => updateNestedField('account.size', e.target.value)}
                        />
                      </div>
                    </div>

                    {/* Main Contact */}
                    <div className="form-section-header">
                      <h3>Main Contact</h3>
                    </div>
                    
                    <div className="form-row">
                      <div className="form-group">
                        <label>Name</label>
                        <input
                          type="text"
                          value={customerData.account.main_contact.name}
                          onChange={(e) => updateNestedField('account.main_contact.name', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Title</label>
                        <input
                          type="text"
                          value={customerData.account.main_contact.title}
                          onChange={(e) => updateNestedField('account.main_contact.title', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Email</label>
                        <input
                          type="email"
                          value={customerData.account.main_contact.email}
                          onChange={(e) => updateNestedField('account.main_contact.email', e.target.value)}
                        />
                      </div>
                    </div>

                    {/* Relationship Status */}
                    <div className="form-section-header">
                      <h3>Relationship Status</h3>
                    </div>
                    
                    <div className="form-row">
                      <div className="form-group">
                        <label>Customer Since</label>
                        <input
                          type="date"
                          value={customerData.account.relationship.customer_since}
                          onChange={(e) => updateNestedField('account.relationship.customer_since', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Deal Stage</label>
                        <select
                          value={customerData.account.relationship.deal_stage}
                          onChange={(e) => updateNestedField('account.relationship.deal_stage', e.target.value)}
                        >
                          <option value="New Customer">New Customer</option>
                          <option value="Onboarding">Onboarding</option>
                          <option value="Implementation">Implementation</option>
                          <option value="Growth">Growth</option>
                          <option value="Mature">Mature</option>
                          <option value="Expansion">Expansion</option>
                          <option value="At Risk">At Risk</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label>Account Health</label>
                        <select
                          value={customerData.account.relationship.account_health}
                          onChange={(e) => updateNestedField('account.relationship.account_health', e.target.value)}
                          className={`health-select health-${customerData.account.relationship.account_health.toLowerCase()}`}
                        >
                          <option value="Excellent">Excellent</option>
                          <option value="Good">Good</option>
                          <option value="Fair">Fair</option>
                          <option value="Poor">Poor</option>
                        </select>
                      </div>
                    </div>

                    {/* Recent Activity Summary */}
                    <div className="form-section-header">
                      <h3>Recent Activity</h3>
                    </div>
                    
                    <div className="activity-summary">
                      <div className="stat-card">
                        <span className="stat-label">Active Users</span>
                        <span className="stat-value">{customerData.recent_activity.product_usage.active_users}</span>
                        <span className="stat-change">{customerData.recent_activity.product_usage.active_users_change}</span>
                      </div>
                      <div className="stat-card">
                        <span className="stat-label">Last Meeting</span>
                        <span className="stat-value">{customerData.recent_activity.meetings[0]?.date || 'N/A'}</span>
                        <span className="stat-subtitle">{customerData.recent_activity.meetings[0]?.type || ''}</span>
                      </div>
                      <div className="stat-card">
                        <span className="stat-label">Open Tickets</span>
                        <span className="stat-value">
                          {customerData.recent_activity.support_tickets.filter(t => t.status.includes('Open')).length}
                        </span>
                        <span className="stat-subtitle">Support Issues</span>
                      </div>
                    </div>

                    {/* Meetings Details */}
                    {customerData.recent_activity.meetings.length > 0 && (
                      <div className="activity-details">
                        <h4>Recent Meetings</h4>
                        {customerData.recent_activity.meetings.map((meeting, idx) => (
                          <div key={idx} className="meeting-card">
                            <div className="meeting-header">
                              <span className="meeting-type">{meeting.type}</span>
                              <span className="meeting-date">{meeting.date}</span>
                            </div>
                            <p className="meeting-summary">{meeting.summary}</p>
                            {meeting.action_items && meeting.action_items.length > 0 && (
                              <div className="action-items">
                                <strong>Action Items:</strong>
                                <ul>
                                  {meeting.action_items.map((item, itemIdx) => (
                                    <li key={itemIdx}>{item}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Product Usage */}
                    <div className="activity-details">
                      <h4>Product Usage Insights</h4>
                      <div className="usage-grid">
                        <div className="usage-section">
                          <h5>Most Used Features</h5>
                          <ul className="feature-list">
                            {customerData.recent_activity.product_usage.most_used_features.map((feature, idx) => (
                              <li key={idx} className="feature-item used">{feature}</li>
                            ))}
                          </ul>
                        </div>
                        <div className="usage-section">
                          <h5>Least Used Features</h5>
                          <ul className="feature-list">
                            {customerData.recent_activity.product_usage.least_used_features.map((feature, idx) => (
                              <li key={idx} className="feature-item unused">{feature}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                      {customerData.recent_activity.product_usage.potential_opportunity && (
                        <div className="opportunity-box">
                          <strong>Opportunity:</strong> {customerData.recent_activity.product_usage.potential_opportunity}
                        </div>
                      )}
                    </div>

                    {/* Support Tickets */}
                    {customerData.recent_activity.support_tickets.length > 0 && (
                      <div className="activity-details">
                        <h4>Support Tickets</h4>
                        <div className="tickets-grid">
                          {customerData.recent_activity.support_tickets.map((ticket, idx) => (
                            <div key={idx} className={`ticket-card ${ticket.status.includes('Open') ? 'open' : 'resolved'}`}>
                              <div className="ticket-header">
                                <span className="ticket-id">{ticket.id}</span>
                                <span className={`ticket-status ${ticket.status.toLowerCase().replace(/\s+/g, '-')}`}>
                                  {ticket.status}
                                </span>
                              </div>
                              <p className="ticket-issue">{ticket.issue}</p>
                              {ticket.priority && (
                                <span className={`ticket-priority priority-${ticket.priority.toLowerCase()}`}>
                                  Priority: {ticket.priority}
                                </span>
                              )}
                              {ticket.resolution && (
                                <p className="ticket-resolution">
                                  <strong>Resolution:</strong> {ticket.resolution}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Sales Rep */}
                    <div className="form-section-header">
                      <h3>Sales Representative</h3>
                    </div>
                    
                    <div className="form-row">
                      <div className="form-group">
                        <label>Name</label>
                        <input
                          type="text"
                          value={customerData.sales_rep.name}
                          onChange={(e) => updateNestedField('sales_rep.name', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Title</label>
                        <input
                          type="text"
                          value={customerData.sales_rep.title}
                          onChange={(e) => updateNestedField('sales_rep.title', e.target.value)}
                        />
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Right Panel - Output */}
          <div className="panel panel-right">
            <div className="output-section">
              <h2>Generated Email</h2>
              
              {!generatedEmail && !loading && (
                <div className="empty-state">
                  <p>Select a company and click "Generate Email" to see the personalized email here.</p>
                </div>
              )}
              
              {loading && (
                <div className="loading-state">
                  <p>Generating personalized email...</p>
                </div>
              )}
              
              {error && (
                <div className="error-message">
                  <h3>Error</h3>
                  <pre>{error}</pre>
                </div>
              )}

              {generatedEmail && !loading && (
                <div className="email-output">
                  {/* Feedback Widget */}
                  <div className="feedback-widget">
                    <div className="feedback-header">
                      <h4>How was this email?</h4>
                    </div>
                    <div className="feedback-buttons">
                      <button 
                        className={`feedback-btn thumbs-up ${feedbackRating === 'up' ? 'active' : ''}`}
                        onClick={() => handleFeedbackRating('up')}
                        title="Good email"
                      >
                        👍
                      </button>
                      <button 
                        className={`feedback-btn thumbs-down ${feedbackRating === 'down' ? 'active' : ''}`}
                        onClick={() => handleFeedbackRating('down')}
                        title="Needs improvement"
                      >
                        👎
                      </button>
                    </div>
                    
                    {/* Comment section - now always visible */}
                    <div className="feedback-comment-section">
                      <textarea
                        className="feedback-comment"
                        placeholder="Share your thoughts about this email... (optional)"
                        value={feedbackComment}
                        onChange={(e) => setFeedbackComment(e.target.value)}
                        rows={3}
                      />
                      <button 
                        className="feedback-submit-btn"
                        onClick={handleFeedbackSubmit}
                        disabled={feedbackSubmitted || !feedbackRating}
                      >
                        {feedbackSubmitted ? '✓ Thank you!' : 'Submit Feedback'}
                      </button>
                    </div>
                  </div>
                  
                  <h3>Subject: {generatedEmail.subject_line}</h3>
                  <pre className="email-body">{generatedEmail.body}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App; 