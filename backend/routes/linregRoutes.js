const express = require('express');
const router = express.Router();

router.post('/', (req, res) => {
  // Your logic for linear regression
  res.send('Linear Regression API');
});

module.exports = router;