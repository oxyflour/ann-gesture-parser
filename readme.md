# ann-gesture-parser
ANN powered gesture parser

### example and usage
see example/index.html for a live example
```javascript
var gesture = new GestureParser({

	// the ANN has an input layer with 20 input nodes, a hidden layer with 30 nodes and
	// an output layer with 8 nodes
	sizes: [20, 30, 8],

	// must be between 0 and 1, smaller is slower
	trainningSpeed: 0.2,

	// max times in a trainning
	maxTrainnningTimes: 1e4,

	// minimal error for trainning
	minTrainningError: 1e-2,

	// values greater than 0.6 will be 1 and less than 0.4 will be 0
	outputThreshold: 0.1,
})

// train the ANN
gesture.add(POSITION_ARRAY, GESTURE_NAME)

assert.equal(gesture.test(POSITION_ARRAY), GESTURE_NAME)
```
