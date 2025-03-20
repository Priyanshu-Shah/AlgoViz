const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');

// Path to the models.json file
const modelsFilePath = path.join(__dirname, '../data/models.json');

// Route to get all models
router.get('/', (req, res) => {
  fs.readFile(modelsFilePath, 'utf8', (err, data) => {
    if (err) {
      console.error('Error reading models.json:', err);
      return res.status(500).json({ error: 'Failed to load models' });
    }
    const models = JSON.parse(data);
    res.json(models);
  });
});

// Route to get a specific model by ID
router.get('/:id', (req, res) => {
  const modelId = req.params.id;

  fs.readFile(modelsFilePath, 'utf8', (err, data) => {
    if (err) {
      console.error('Error reading models.json:', err);
      return res.status(500).json({ error: 'Failed to load models' });
    }
    const models = JSON.parse(data);
    const model = models.find(m => m.id === modelId);

    if (!model) {
      return res.status(404).json({ error: 'Model not found' });
    }

    res.json(model);
  });
});

module.exports = router;