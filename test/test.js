var assert = require('assert'),
	GestureParser = require('../index'),
	trainStore = require('./trainStore.json'),
	testStore = require('./testStore.json')

describe('ann parser', function() {

	it('should train the network and pass tests', function() {

		var gesture = new GestureParser()

		this.timeout(120000)
		console.log('training the network (may take a few minutes...)')

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
