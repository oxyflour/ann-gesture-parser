var TAK = k => a => a[k],
	ADD = (a, b) => a + b,
	SUB = (a, b) => a - b,
	MUL = (a, b) => a * b

function pulse(x) {
	return 1 / (1 + Math.exp(-x))
}

function hypot(x, y) {
	return Math.sqrt(x * x + y * y)
}

function interp(min, max, f) {
	return min * (1 - f) + max * f
}

function random(min, max) {
	return interp(+min === min ? min : 0, +max === max ? max : 1, Math.random())
}

function vector(size, fn) {
	return Array(size).fill(0).map(fn ? fn : (_ => 0))
}

function pDist(p1, p2) {
	return hypot(p1.x - p2.x, p1.y - p2.y)
}

function pDiff(p1, p2) {
	return { x:p1.x - p2.x, y:p1.y - p2.y }
}

function pInterp(p1, p2, f) {
	return { x:interp(p1.x, p2.x, f), y:interp(p1.y, p2.y, f) }
}

Array.prototype.vAs = function(val) {
	return Array(this.length).fill(val)
}

Array.prototype.vTake = function(key) {
	return this.map(TAK(key))
}

Array.prototype.vSum = function() {
	return this.reduce(ADD, 0)
}

Array.prototype.vMap = function(func, arr) {
	var ret = Array(this.length),
		that = Array.isArray(arr) ? arr : this.vAs(arr)
	for (var i = 0, n = ret.length; i < n; i ++)
		ret[i] = func(this[i], that[i])
	return ret
}

Array.prototype.vAdd = function(arr) {
	return this.vMap(ADD, arr)
}

Array.prototype.vSub = function(arr) {
	return this.vMap(SUB, arr)
}

Array.prototype.vMul = function(arr) {
	return this.vMap(MUL, arr)
}

function resortPoints(pts, dist) {
	var ret = [ pts[0] ]
	pts.slice(1).forEach(pt => {
		if (pDist(ret[ret.length - 1], pt) > dist)
			ret.push(pt)
	})
	return ret
}

function interpPoints(pts, segments) {
	var eachLen = pts.slice(1).map((_, i) => pDist(pts[i], pts[i + 1])),
		segLen = eachLen.vSum() / segments,
		startLen = 0,
		interpPoints = [ ]
	eachLen.forEach((len, i) => {
		for (var start = startLen; start < len + 1e-3; start += segLen)
			interpPoints.push(pInterp(pts[i], pts[i + 1], start / len))
		startLen = start - len
	})
	return interpPoints
}

function normailizePoints(pts) {
	var left = 1/0, right = -1/0,
		top = 1/0, bottom = -1/0
	pts.forEach(pt => {
		left = Math.min(left, pt.x)
		right = Math.max(right, pt.x)
		top = Math.min(top, pt.y)
		bottom = Math.max(bottom, pt.y)
	})
	var size = Math.max(right - left, bottom - top)
	return pts.map(pt => ({
		x: (pt.x - left) / size,
		y: (pt.y - top) / size,
	}))
}

function getOutput(weight, bias, input) {
	var result = [ input ]
	for (var i = 0; i < weight.length; i ++) {
		result[i + 1] = weight[i].map(c => c.vMul(result[i]).vSum())
			.vAdd(bias[i]).map(pulse)
	}
	return result
}

function adjustNetwork(weight, bias, input, excepted, speed) {
	var result = getOutput(weight, bias, input),
		output = result[result.length - 1],
		delta = excepted.vSub(output),
		error = output.vMul(output.vAs(1).vSub(output)).vMul(delta)
	for (var i = weight.length - 1; i >= 0; i --) {
		output = result[i]
		weight[i] = weight[i].map((m, k) => m.vAdd(output.vMul(error[k] * speed)))
		bias[i] = bias[i].vAdd(error.vMul(speed))
		delta = output.map((_, j) => weight[i].vTake(j).vMul(error).vSum())
		error = output.vMul(output.vAs(1).vSub(output)).vMul(delta)
	}
	error = excepted.vSub(result[result.length - 1])
	return error.vMul(error).vSum() / error.length
}

function makeInput(pts, size) {
	pts = normailizePoints(interpPoints(resortPoints(pts, 5), size / 2))
	return pts.slice(1)
		.map((p, i) => pDiff(pts[i + 1], pts[i]))
		.reduce((s, c) => s.concat(c.x, c.y), [ ])
}

function makeExcepted(index, size) {
	var bits = index.toString(2).split('').map(parseFloat)
	return Array(size).fill(0).concat(bits).slice(-size)
}

function getIndex(output, threshold) {
	var bits = output.map(v => v > 0.5 + threshold ? 1 : (v < 0.5 - threshold ? 0 : 'x'))
	return parseInt(bits.join(''), 2)
}

function AnnGestureParser(opts) {
	this.opts = opts || { }
	this.opts.sizes 				= this.opts.sizes 				|| [20, 30, 8]
	this.opts.trainningSpeed 		= this.opts.trainningSpeed 		|| 0.2
	this.opts.maxTrainnningTimes 	= this.opts.maxTrainnningTimes 	|| 1e4
	this.opts.minTrainningError 	= this.opts.minTrainningError 	|| 1e-2
	this.opts.outputThreshold 		= this.opts.outputThreshold 	|| 0.1

	var sizes = this.opts.sizes
	this.inputSize = sizes[0]
	this.outputSize = sizes[sizes.length - 1]

	this.networkWeight = sizes.slice(1).map((_, i) =>
		vector(sizes[i + 1], _ => vector(sizes[i], _ => random(-1, 1))))
	this.networkBias = sizes.slice(1).map(v => vector(v, _ => random(0, 1)))
	this.trainStore = { /* ret: { inputs:[], index:Number } */ }
}

AnnGestureParser.prototype.update = function() {
	var weight = this.networkWeight,
		bias = this.networkBias,
		speed = this.opts.trainningSpeed,
		rets = Object.keys(this.trainStore),
		times = 0, error = 1/0

	while (times ++ < this.opts.maxTrainnningTimes &&
			error > this.opts.minTrainningError) {
		error = 0
		rets.forEach(ret => {
			var store = this.trainStore[ret],
				excepted = makeExcepted(store.index, this.outputSize)
			store.inputs.forEach(input => {
				error += adjustNetwork(weight, bias, input, excepted, speed)
			})
		})
	}

	return error
}

AnnGestureParser.prototype.add = function(pts, ret) {
	var trainStore = this.trainStore,
		input = makeInput(pts, this.inputSize),
		index = Math.max.apply(null,
			Object.keys(trainStore).map(k => trainStore[k].index).concat(0)),
		store = trainStore[ret] ||
			(trainStore[ret] = { index: index + 1, inputs: [ ] })

	store.inputs.push(input)

	var error = this.update()
	if (error > this.opts.minTrainningError)
		console.warn('result does not converge, error', error)

	return this.test(pts)
}

AnnGestureParser.prototype.test = function(pts) {
	var weight = this.networkWeight,
		bias = this.networkBias,
		trainStore = this.trainStore,
		input = makeInput(pts, this.inputSize),
		result = getOutput(weight, bias, input),
		output = result[result.length - 1],
		index = getIndex(output, this.opts.outputThreshold)

	return Object.keys(trainStore).filter(k => trainStore[k].index === index)[0]
}

if (typeof module !== 'undefined')
	module.exports = AnnGestureParser