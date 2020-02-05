//
//  FileTests.swift
//  FastNeuralNetworkTests
//
//  Created by Korben Rusek on 2/4/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import XCTest
@testable import FastNeuralNetwork

class FileTests: XCTestCase {

    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    let sevenString = """
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000001111110000000000000000
0000001111111111111111000000
0000001111111111111111000000
0000000000011111111111000000
0000000000000000001111000000
0000000000000000011110000000
0000000000000000011110000000
0000000000000000111100000000
0000000000000000111100000000
0000000000000001111000000000
0000000000000001110000000000
0000000000000011110000000000
0000000000000111100000000000
0000000000001111100000000000
0000000000001111000000000000
0000000000011111000000000000
0000000000011110000000000000
0000000000111110000000000000
0000000000111110000000000000
0000000000111100000000000000
0000000000000000000000000000
"""

    let expectedTwo = """
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000111111100000000000
0000000001111111110000000000
0000000011111111110000000000
0000000111111011110000000000
0000000111100011110000000000
0000000011000011110000000000
0000000000000111110000000000
0000000000001111100000000000
0000000000001111000000000000
0000000000011111000000000000
0000000000111110000000000000
0000000000111100000000000000
0000000001111100000000000000
0000000011111000000000000000
0000000011111000000000000000
0000000011110000000000000000
0000000011111111101111111110
0000000011111111111111111110
0000000011111111111111111110
0000000001111111111110000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
"""

    func testStreamCreate() {
        guard let stream = FastNeuralNetwork.DataStream.images(.t10k) else {
            XCTFail()
            return
        }

        XCTAssertEqual(stream.readInt(), 2051)
        XCTAssertEqual(stream.readInt(), 10000)
        let rows = stream.readInt()
        let cols = stream.readInt()
        let foundSeven = readStrings(stream: stream, rows: rows, cols: cols)
        XCTAssertEqual(foundSeven, sevenString)
        XCTAssertEqual(readStrings(stream: stream, rows: rows, cols: cols), expectedTwo)
    }

    func readStrings(stream: FastNeuralNetwork.DataStream, rows: Int32, cols: Int32) -> String {
        let strings = (0..<rows).map { _ in
            stream.readBytes(count: Int(cols)).map { $0 != 0 ? "1" : "0" }.joined()
        }
        return strings.joined(separator: "\n")

    }

    func testLabelsCreate() {
        guard let stream = FastNeuralNetwork.DataStream.labels(.t10k) else {
            XCTFail()
            return
        }

        XCTAssertEqual(stream.readInt(), 2049)
        XCTAssertEqual(stream.readInt(), 10000)
        XCTAssertEqual(stream.readByte(), 7)
        XCTAssertEqual(stream.readByte(), 2)
    }

}
