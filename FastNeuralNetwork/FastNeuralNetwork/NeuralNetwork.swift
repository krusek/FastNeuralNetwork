//
//  NeuralNetwork.swift
//  FastNeuralNetwork
//
//  Created by Korben Rusek on 1/28/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import Foundation
import Matrix

public typealias Generator = () -> Double

extension Matrix {
    static func random(rows: Int, cols: Int) -> Matrix {
        let generator = { Double.random(in: 0...1) }
        return random(rows: rows, cols: cols, generator: generator)
    }
    static func random(rows: Int, cols: Int, generator: Generator) -> Matrix {
        let data: [[Double]] = (0..<rows).map { _ in (0..<cols).map { _ in generator() } }
        return Matrix(data)
    }
}

struct GeneratorData {
    let weight: Generator
    let bias: Generator
    let sigmoid: (Matrix) -> Matrix

    static var defaultValue: GeneratorData {
        return GeneratorData(weight: { Double.random(in: 0...1) }, bias: { Double.random(in: 0...1) }, sigmoid: { (1.0 / ($0.exp() + 1.0)) })
    }
}

public class NeuralNetwork {
    let biases: [Matrix]
    let weights: [Matrix]
    let sigmoid: (Matrix) -> Matrix

    init(sizes: [Int], generator: GeneratorData) {
        let biases = sizes.dropFirst().map { Matrix.random(rows: $0, cols: 1, generator: generator.bias) }
        let weights = zip(sizes.dropLast(), sizes.dropFirst()).map { Matrix.random(rows: $1, cols: $0, generator: generator.weight) }
        self.sigmoid = generator.sigmoid
        self.biases = biases
        self.weights = weights

    }

    public convenience init(sizes: [Int]) {
        self.init(sizes: sizes, generator: GeneratorData.defaultValue)
    }

    public init(biases: [Matrix], weights: [Matrix]) {
        self.biases = biases
        self.weights = weights
        self.sigmoid = GeneratorData.defaultValue.sigmoid
    }

    public func feedForward(_ a: Matrix) -> Matrix {
        var last = a
        var current = a
        for (b, w) in zip(biases, weights) {
            current = (w <*> last) + b
            current = sigmoid(current)
            let temp = last
            last = current
            current = temp
        }
        return last
    }
}

extension NeuralNetwork {
    public func evaluate(_ test_data: [(PhotoData, [Double])]) -> Int {
       var correct = 0
       for (x, y) in test_data {
            print(x)
            let xx = Matrix(x.map { Double($0) }, isColumnVector: true)
            let forward = self.feedForward(xx)
            let fx = forward.max().2
            if let my = mxIndex(y), fx == my {
               correct += 1
           }
       }
       return correct
   }

    private func mxIndex<X: Comparable, C: Collection>(_ array: C) -> Int? where C.Element == X, C.Index == Int {
        guard let mx = array.max(), let first = array.firstIndex(of: mx) else { return nil }
        return first
    }
}
