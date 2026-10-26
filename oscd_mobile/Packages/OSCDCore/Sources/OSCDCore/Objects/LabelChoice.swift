import Foundation

/// Which classifier label names an object (R34). Vision lists a label's ancestors with almost its confidence
/// ("recreation" 0.81, "sport" 0.80, "ball" 0.80), so the top label alone is often a category, not the object.
public enum LabelChoice {
    /// Category labels that name no particular object.
    public static let generic: Set<String> = [
        "recreation", "sport", "sports_equipment", "structure", "material", "object", "indoor", "outdoor", "textile",
        "furniture", "consumer_electronics", "tool", "container", "art", "people", "animal", "plant", "food", "clothing",
    ]

    /// Labels within this much of the top confidence count as tied with it.
    public static let tie: Float = 0.02

    /// Among the labels at or above `minConfidence` whose confidence is within `tie` of the highest, the first in the
    /// given (Vision's) order that is not in `generic`; when every tied label is generic, the highest (the first of
    /// equal highs). Nil when no label reaches `minConfidence`. Non-finite confidences are ignored.
    public static func pick(_ labels: [(String, Float)], minConfidence: Float = 0) -> (String, Float)? {
        let candidates = labels.filter { $0.1.isFinite && $0.1 >= minConfidence }
        guard let top = candidates.max(by: { $0.1 < $1.1 }) else { return nil }
        let tied = candidates.filter { $0.1 >= top.1 - tie }
        return tied.first { !isGeneric($0.0) } ?? candidates.first { $0.1 == top.1 }
    }

    /// True for a category label; an identifier with alternatives ("sunglasses,shades") is judged by its first.
    public static func isGeneric(_ identifier: String) -> Bool {
        let first = identifier.split(separator: ",").first.map(String.init) ?? identifier
        return generic.contains(first.trimmingCharacters(in: .whitespaces).lowercased())
    }
}
