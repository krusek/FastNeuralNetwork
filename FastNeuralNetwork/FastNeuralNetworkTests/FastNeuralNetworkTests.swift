//
//  FastNeuralNetworkTests.swift
//  FastNeuralNetworkTests
//
//  Created by Korben Rusek on 1/28/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import XCTest
@testable import Matrix
@testable import FastNeuralNetwork

class FastNeuralNetworkTests: XCTestCase {
    func testWeightAndBiasSizesWork() {
        let sizes = [2, 3, 4]
        let network = NeuralNetwork(sizes: sizes)
        XCTAssertEqual(sizes[1], network.biases[0].rows)
        let _ = network.weights[0].transpose() <*> network.biases[0]
    }

    func testSimpleFeedforward() {
        let generator = GeneratorData(weight: { 2.0 }, bias: { 1.0 }, sigmoid: { $0 + 1.0 } )
        let sizes = [2, 3, 4]
        let network = NeuralNetwork(sizes: sizes, generator: generator)
        let a = network.feedForward(Matrix([1.0, 2.0], isColumnVector: true))
        XCTAssertEqual(a.grid, [50.0, 50.0, 50.0, 50.0])
    }

    func testSimpleFeedforward2() {
        let generator = GeneratorData(weight: { 2.0 }, bias: { 1.0 }, sigmoid: { $0 + 1.0 } )
        let sizes = [3, 2, 1]
        let network = NeuralNetwork(sizes: sizes, generator: generator)
        let a = network.feedForward(Matrix([1.0, 2.0, 1.0], isColumnVector: true))
        XCTAssertEqual(a.grid, [42.0])
    }

    func testEvaluate() {
        var x = 0.0
        let generator = GeneratorData(weight: {
            x += 1
            return x
        }, bias: {
            x += 1
            return x
        }, sigmoid: { $0 + 1.0 } )
        let sizes = [13, 20, 11]
        let network = NeuralNetwork(sizes: sizes, generator: generator)
        let data = (0..<13).map { UInt8($0) }
        for ix in 0..<13 {
            let a = (0..<13).map { $0 == ix ? 1.0 : 0.0 }
            let e = network.evaluate([(data, a)])
            XCTAssertEqual(e, ix == 10 ? 1 : 0)
        }
    }
}
