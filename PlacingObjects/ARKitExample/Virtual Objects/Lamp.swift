/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The virtual lamp.
*/

import Foundation
import ARKit

@available(iOS 11.0, *)
class Lamp: VirtualObject {
	
	override init() {
		super.init(modelName: "lamp", fileExtension: "scn", thumbImageFilename: "lamp", title: "Lamp")
	}
	
	required init?(coder aDecoder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}
}
