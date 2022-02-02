@0xcdcceafdccee39a2;

# document level

struct SharedListInt {
    # for spans: ref into Document.spans, or -1 if no span
    values :union {
        int8 @0 :List(Int8);
        int16 @1 :List(Int16);
        int32 @2 :List(Int32);
    }
}

struct CodeData {
    code @0 :Int32;  # ref into Archive.codes
    spans @1 :SharedListInt;  # sorted numerically, i.e. by span id
    labels :group {
        values :union {  # ref into Tagger.values
            none @2 :Void;
            uint8 @3 :List(UInt8);
            uint16 @4 :List(UInt16);
            uint32 @5 :List(UInt32);
        }
        scores :union {
            none @6 :Void;
            float32 @7 :List(Float32);
            float64 @8 :List(Float64);
        }
        groups :union {  # offsets into values, scores
            none @9 :Void;
            uint8 @10 :List(UInt8);
            uint16 @11 :List(UInt16);
            uint32 @12 :List(UInt32);
        }
    }
    parents :union {
        none @13 :Void;
        indices @14 :SharedListInt;
    }
}

struct Document {
    text @0 :Data;  # utf8 encoded
    meta @1 :Text;  # json

    tags @2 :List(CodeData);  # sorted by CodeData.code

    # spans:
    # * byte index into Document.text
    # * sorted by (Span.start, Span.start - Span.end))

    starts :union {
        uint8 @3 :List(UInt8);
        uint16 @4 :List(UInt16);
        uint32 @5 :List(UInt32);
        uint64 @6 :List(UInt64);
    }
    lens :union {
        uint8 @7 :List(UInt8);
        uint16 @8 :List(UInt16);
        uint32 @9 :List(UInt32);
        uint64 @10 :List(UInt64);
    }
}

# archive level

struct Tagger {
    guid @0 :Text;
    signature @1 :Text;
    codes @2 :List(Int32);  # ref into Archive.codes
}

struct Code {
    tagger @0 :Int32;  # ref into Archive.taggers
    name @1 :Text;
    values @2 :List(Text);
}

struct DocumentRef {
    start @0 :UInt64;
    end @1 :UInt64;
}

struct Archive {
    version @0 :UInt64;
    taggers @1 :List(Tagger);
    codes @2 :List(Code);  # random order
    documents @3 :List(DocumentRef);
}
