const express = require('express');
const router = express.Router();
const models = require('../data/models.json');
const kmeansService = require('../services/kmeansService');
const decisionTreeService = require('../services/decisionTreeService');

router.get('/', (req, res) => {
  res.json(models);
});

// Serve model-specific data
router.get('/:id', (req, res) => {
  const model = models.find(model => model.id === req.params.id);
  if (!model) {
    return res.status(404).json({ error: 'Model not found' });
  }
  res.json(model);
});

module.exports = router;
