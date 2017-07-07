/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The virtual chair.
*/

import Foundation

@available(iOS 11.0, *)
class Chair: VirtualObject {
	
	override init() {
		super.init(modelName: "chair", fileExtension: "scn", thumbImageFilename: "chair", title: "Chair")
	}
	
	required init?(coder aDecoder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}
}
