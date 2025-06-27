import ReviewForm from './Components/Reviewform.jsx';
import Header from './Components/Header.jsx';
import Footer from './Components/Footer.jsx';
import background from './assets/background.jpg';
import SentimentResult from './Components/SentimentResult.jsx';
import React, { useState } from 'react';
import './App.css'; // Import custom CSS for additional styles
function App() {
  const [sentiment, setSentiment] = useState("");
  return (
    <div className="d-flex flex-column min-vh-100 bg-img" >
      <Header />
      
      {/* Main content area: grows and centers content */}
      <main className="flex-grow-1 d-flex justify-content-center align-items-center bg-overlay">
        <ReviewForm setSentiment={setSentiment} />
        <br />
        <SentimentResult sentiment={sentiment} />
      </main>


      <Footer />
    </div>
  );
}

export default App;
