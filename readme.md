# ann-gesture-parser
ANN powered gesture parser

### example and usage
see example/index.html for a live example
```javascript
var gesture = new GestureParser({

	// the ANN has an input layer with 20 nodes, a hidden layer with 30 nodes and
	// an output layer with 8 nodes
	sizes: [20, 30, 8],

	// must be between 0 and 1, smaller is slower
	trainingSpeed: 0.5,

	// max loops in one training
	maxTrainingTimes: 1e4,

	// minimal error to end training
	minTrainingError: 1e-3,

	// output values greater than (0.5 + threshold) will be 1,
	// and those less than (0.5 - threshold) will be 0
	outputThreshold: 0.1,
})

// train the ANN
gesture.add(POSITION_ARRAY, GESTURE_NAME)

assert.equal(gesture.test(POSITION_ARRAY), GESTURE_NAME)
```
