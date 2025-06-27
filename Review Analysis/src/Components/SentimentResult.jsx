function SentimentResult({ sentiment }) {
  if (!sentiment) return null;

  let colorClass = "info";
  if (sentiment === "positive") colorClass = "success";
  else if (sentiment === "negative") colorClass = "danger";
  else colorClass = "secondary";

  return (
    <div className={`alert alert-${colorClass} text-center`}>
      <strong>Detected Sentiment:</strong> {sentiment}
    </div>
  );
}

export default SentimentResult;
