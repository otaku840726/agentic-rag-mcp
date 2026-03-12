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
        public string? fqn { get; set; }              // fully qualified name
        public string node_type { get; set; } = "";   // class, method, property, field, etc.
        public string content { get; set; } = "";
        public int start_byte { get; set; }
        public int end_byte { get; set; }
        public int start_line { get; set; }           // 0-indexed
        public int end_line { get; set; }             // 0-indexed

        // ── Semantic enrichment fields (emitted directly by RoslynAnalyzer) ──
        public string? file_type { get; set; }        // controller, service, repository, entity, config, middleware
        public string? entry_point_type { get; set; } // api, queue, job
        public string? http_method { get; set; }      // GET, POST, PUT, DELETE, PATCH
        public string? http_path { get; set; }        // /api/v1/users
        public bool? auth_required { get; set; }
        public string? auth_roles { get; set; }
        public string? operation_type { get; set; }   // READ, WRITE, DELETE
        public string? table_name { get; set; }
        public bool? makes_http_call { get; set; }
        public bool? is_test { get; set; }
        public bool? is_deprecated { get; set; }
        public string? visibility { get; set; }       // public, private, protected, internal
        public bool? is_static { get; set; }
        public bool? is_abstract { get; set; }

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
