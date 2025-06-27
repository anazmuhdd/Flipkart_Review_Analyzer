import React, { useState } from "react";
import axios from "axios";

function ReviewForm({ setSentiment }) {
  const [review, setreview] = useState("");
  const [error, seterror] = useState("");

  const handlesubmit = async (e) => {
    e.preventDefault();
    seterror("");

    try {
      const response = await axios.post("http://localhost:5000/predict", {
        review: review,
      });
      console.log("Predicted sentiment:", response.data.sentiment);
      setSentiment(response.data.sentiment);

      // You could pass it to a parent via props or use local state
    } catch (err) {
      seterror("Error occurred while analyzing sentiment");
    }
  };

  return (
    <form className="w-75 w-md-50 text-center" onSubmit={handlesubmit}>
      <textarea
        name="review"
        id="review"
        placeholder="Write your review here..."
        className="form-control mb-3 w-50 mx-auto"
        rows="4"
        style={{ backgroundColor: "white", color: "black" }}
        value={review}
        onChange={(e) => setreview(e.target.value)}
        required
      ></textarea>
      <button className="btn btn-primary" type="submit">
        Analyze Sentiment
      </button>
      {error && (
        <div className="alert alert-danger mt-3">{error}</div>
      )}
    </form>
  );
}

export default ReviewForm;
