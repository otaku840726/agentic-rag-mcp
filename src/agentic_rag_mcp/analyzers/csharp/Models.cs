using System.Collections.Generic;

namespace RoslynAnalyzer
{
    public class AnalysisResult
    {
        public string file_path { get; set; } = "";
        public string language { get; set; } = "csharp";
        public List<Symbol> symbols { get; set; } = new List<Symbol>();
        public List<Relationship> relationships { get; set; } = new List<Relationship>();
        public object? raw_ast { get; set; } = null;
    }

    public class Symbol
    {
        public string name { get; set; } = "";
        public string node_type { get; set; } = ""; // class, method, etc.
        public string content { get; set; } = "";
        public int start_byte { get; set; }
        public int end_byte { get; set; }
        public int start_line { get; set; } // 0-indexed
        public int end_line { get; set; }   // 0-indexed
        public Dictionary<string, object> metadata { get; set; } = new Dictionary<string, object>();
    }

    public class Relationship
    {
        public string source { get; set; } = "";
        public string target { get; set; } = "";
        public string type { get; set; } = ""; // calls, inherits, implements
        public Dictionary<string, object> metadata { get; set; } = new Dictionary<string, object>();
    }
}
