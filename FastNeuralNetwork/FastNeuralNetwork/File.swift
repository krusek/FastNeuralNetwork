//
//  File.swift
//  FastNeuralNetwork
//
//  Created by Korben Rusek on 2/4/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import Foundation

public typealias PhotoData = [UInt8]

public enum DataName {
    case t10k, train

    public var labelFilename: String {
        switch self {
        case .t10k:
            return "t10k-labels-idx1-ubyte"
        case .train:
            return "train-labels-idx1-ubyte"
        }
    }

    public var imageFilename: String {
        switch self {
        case .t10k:
            return "t10k-images-idx3-ubyte"
        case .train:
            return "train-images-idx3-ubyte"
        }
    }
}

public class DataStream {
    let data: Data
    var index: Data.Index
    init?(_ name: String) {
        let path = "/Users/korbenrusek/Documents/code/neural/neuralnetwork"
        let filePath = "\(path)/data/\(name)"
        print(filePath)
        let exists = FileManager.default.fileExists(atPath: filePath)
        print("exists: \(exists)")
        if let data = FileManager.default.contents(atPath: filePath) {
            self.data = data
            self.index = data.startIndex
        } else {
            return nil
        }
    }

    public static func labels(_ name: DataName) -> DataStream? {
        return DataStream(name.labelFilename)
    }

    public static func images(_ name: DataName) -> DataStream? {
        return DataStream(name.imageFilename)
    }

    public func readInt() -> Int32 {
        let pointer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 4)
        let end = index.advanced(by: 4)
        data.copyBytes(to: pointer, from: Range<Data.Index>(uncheckedBounds: (index, end)))
        var int: Int32 = 0
        for item in pointer {
            int = (int << 8) | Int32(item)
        }
        self.index = end
        return int
    }

    public func readByte() -> UInt8 {
        let pointer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 1)
        let end = index.advanced(by: 1)
        data.copyBytes(to: pointer, from: Range<Data.Index>(uncheckedBounds: (index, end)))
        self.index = end
        return pointer.first ?? 0
    }

    public func readBytes(count: Int) -> [UInt8] {
        let pointer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: count)
        let end = index.advanced(by: count)
        data.copyBytes(to: pointer, from: Range<Data.Index>(uncheckedBounds: (index, end)))
        self.index = end
        return Array(pointer)
    }

    public static func readTrainingData(_ type: DataName, _ limit: Int?) throws -> [(PhotoData, [Double])] {
        guard let images = DataStream.images(type), let labels = DataStream.images(type) else {
            throw "Files not found."
        }

        _ = images.readInt()
        _ = labels.readInt()
        let size1 = images.readInt()
        let size2 = labels.readInt()
        guard size1 == size2 else {
            throw "Files of different sizes! images: \(size1), labels: \(size2)"
        }

        let rows = images.readInt()
        let cols = images.readInt()
        let range: Range<Int32>
        if let limit = limit {
            range = (0..<min(size1, Int32(limit)))
        } else {
            range = 0..<size1
        }
        return range.map { (_) -> (PhotoData, [Double]) in
            let data = images.readBytes(count: Int(rows * cols))
            let expected = labels.readByte()
            return  (data, intToArray(expected))
        }
    }
}

extension String: Error {}

func intToArray(_ int: UInt8) -> [Double] {
    var array = Array<Double>(repeating: 0.0, count: 10)
    guard int < 10 && int >= 0 else { return array }
    array[Int(int)] = 1.0
    return array
}

