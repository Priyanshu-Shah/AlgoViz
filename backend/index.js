const express = require('express');
const cors = require('cors');
const modelRoutes = require('./routes/modelRoutes');

const app = express();
app.use(cors());
app.use(express.json());

app.use('/models', modelRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
