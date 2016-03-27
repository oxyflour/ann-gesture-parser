var assert = require('assert'),
	GestureParser = require('../index'),
	trainStore = require('./trainStore.json'),
	testStore = require('./testStore.json')

describe('testing parser (may take a few minutes...)', function() {

	it('should train the network and pass tests', function() {
		this.timeout(60000)

		var gesture = new GestureParser()

		Object.keys(trainStore).forEach(key => {
			trainStore[key].forEach(pts => {
				gesture.add(pts, key)
			})
		})

		Object.keys(testStore).forEach(key => {
			testStore[key].forEach(pts => {
				assert.equal(gesture.test(pts), key)
			})
		})
	})

})
