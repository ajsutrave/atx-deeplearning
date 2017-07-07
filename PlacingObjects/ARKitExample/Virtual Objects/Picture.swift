/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 The virtual candle.
 */

import Foundation
import SceneKit

@available(iOS 11.0, *)
class Picture: VirtualObject, ReactsToScale {
    
    
    
    override init() {
        let boxGeometry = SCNBox(width: 10.0, height: 10.0, length: 10.0, chamferRadius: 1.0)
        super.init(aGeometry: boxGeometry)
        
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func reactToScale() {
        // Update the size of the flame
        let flameNode = self.childNode(withName: "flame", recursively: true)
        let particleSize: Float = 0.018
        flameNode?.particleSystems?.first?.reset()
        flameNode?.particleSystems?.first?.particleSize = CGFloat(self.scale.x * particleSize)
    }
}

