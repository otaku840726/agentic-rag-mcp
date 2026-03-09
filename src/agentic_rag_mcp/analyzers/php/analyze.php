<?php
/**
 * PHP AST Analyzer using nikic/PHP-Parser
 *
 * Usage:  php analyze.php <file_path>
 * Output: JSON matching AnalysisResult contract consumed by agentic-rag-mcp
 *
 * Output format:
 * {
 *   "file_path": "...",
 *   "language": "php",
 *   "symbols": [ { "fqn", "name", "kind", "namespace", "start_line", "end_line",
 *                  "content", "visibility", "is_static", "is_abstract" }, ... ],
 *   "relationships": [ { "type", "source", "target" }, ... ],
 *   "raw_ast": null
 * }
 */

require_once __DIR__ . '/vendor/autoload.php';

use PhpParser\Error;
use PhpParser\Node;
use PhpParser\Node\Stmt;
use PhpParser\NodeTraverser;
use PhpParser\NodeVisitorAbstract;
use PhpParser\ParserFactory;
use PhpParser\PrettyPrinter;

// ── Entry point ───────────────────────────────────────────────────────────────

if ($argc < 2) {
    fwrite(STDERR, "Usage: php analyze.php <file_path>\n");
    exit(1);
}

$filePath = $argv[1];

if (!file_exists($filePath)) {
    fwrite(STDERR, "File not found: $filePath\n");
    exit(1);
}

$code = file_get_contents($filePath);
if ($code === false) {
    fwrite(STDERR, "Cannot read file: $filePath\n");
    exit(1);
}

// ── Parse ────────────────────────────────────────────────────────────────────

$parser = (new ParserFactory())->createForNewestSupportedVersion();

try {
    $ast = $parser->parse($code);
} catch (Error $e) {
    // Return empty result on parse error rather than crashing
    outputResult($filePath, [], []);
    exit(0);
}

if ($ast === null) {
    outputResult($filePath, [], []);
    exit(0);
}

// ── Collect symbols and relationships ────────────────────────────────────────

$lines     = explode("\n", $code);
$symbols   = [];
$rels      = [];
$namespace = '';

/**
 * Extract the full text content for a node (by line range).
 */
function nodeContent(array $lines, Node $node): string {
    $start = ($node->getStartLine() ?? 1) - 1;
    $end   = ($node->getEndLine()   ?? 1) - 1;
    $start = max(0, $start);
    $end   = min(count($lines) - 1, $end);
    return implode("\n", array_slice($lines, $start, $end - $start + 1));
}

/**
 * Get a visibility string from flags.
 */
function visibilityStr(int $flags): string {
    if ($flags & Stmt\Class_::MODIFIER_PRIVATE)   return 'private';
    if ($flags & Stmt\Class_::MODIFIER_PROTECTED) return 'protected';
    return 'public';
}

/**
 * Recursively walk the AST, collecting symbols and relationships.
 */
function walkStmts(
    array  $stmts,
    array  $lines,
    string $currentNamespace,
    string $parentFqn,
    array  &$symbols,
    array  &$rels
): void {
    foreach ($stmts as $stmt) {
        if ($stmt instanceof Stmt\Namespace_) {
            $ns = $stmt->name ? $stmt->name->toString() : '';
            walkStmts($stmt->stmts, $lines, $ns, '', $symbols, $rels);
            continue;
        }

        if ($stmt instanceof Stmt\Use_) {
            // use Foo\Bar\Baz; or use Foo\Bar\Baz as Alias;
            foreach ($stmt->uses as $use) {
                $rels[] = [
                    'type'   => 'IMPORTS',
                    'source' => $parentFqn ?: $currentNamespace,
                    'target' => $use->name->toString(),
                ];
            }
            continue;
        }

        // ── Class / Interface / Trait ─────────────────────────────────────
        if ($stmt instanceof Stmt\Class_
            || $stmt instanceof Stmt\Interface_
            || $stmt instanceof Stmt\Trait_) {

            $name = $stmt->name ? $stmt->name->toString() : 'anonymous';
            $fqn  = $currentNamespace ? "$currentNamespace\\$name" : $name;

            if ($stmt instanceof Stmt\Class_) {
                $kind = 'class';
                // extends → INHERITS
                if ($stmt->extends) {
                    $rels[] = [
                        'type'   => 'INHERITS',
                        'source' => $fqn,
                        'target' => $stmt->extends->toString(),
                    ];
                }
                // implements → IMPLEMENTS
                foreach ($stmt->implements as $iface) {
                    $rels[] = [
                        'type'   => 'IMPLEMENTS',
                        'source' => $fqn,
                        'target' => $iface->toString(),
                    ];
                }
                // use trait → TRAIT_USE
                foreach ($stmt->stmts as $member) {
                    if ($member instanceof Stmt\TraitUse) {
                        foreach ($member->traits as $trait) {
                            $rels[] = [
                                'type'   => 'TRAIT_USE',
                                'source' => $fqn,
                                'target' => $trait->toString(),
                            ];
                        }
                    }
                }
            } elseif ($stmt instanceof Stmt\Interface_) {
                $kind = 'interface';
                foreach ($stmt->extends as $parent) {
                    $rels[] = [
                        'type'   => 'INHERITS',
                        'source' => $fqn,
                        'target' => $parent->toString(),
                    ];
                }
            } else {
                $kind = 'trait';
            }

            $symbols[] = [
                'fqn'         => $fqn,
                'name'        => $name,
                'kind'        => $kind,
                'namespace'   => $currentNamespace,
                'start_line'  => $stmt->getStartLine() ?? 0,
                'end_line'    => $stmt->getEndLine()   ?? 0,
                'content'     => nodeContent($lines, $stmt),
                'node_type'   => $kind,          // alias used by Qdrant payload
                'is_abstract' => ($stmt instanceof Stmt\Class_ && $stmt->isAbstract()),
                'is_static'   => false,
                'visibility'  => 'public',
            ];

            // Walk class/interface/trait body for methods
            walkStmts($stmt->stmts ?? [], $lines, $currentNamespace, $fqn, $symbols, $rels);
            continue;
        }

        // ── Methods ───────────────────────────────────────────────────────
        if ($stmt instanceof Stmt\ClassMethod) {
            $methodName = $stmt->name->toString();
            $fqn = $parentFqn ? "$parentFqn::$methodName" : $methodName;

            $symbols[] = [
                'fqn'         => $fqn,
                'name'        => $methodName,
                'kind'        => 'method',
                'namespace'   => $currentNamespace,
                'start_line'  => $stmt->getStartLine() ?? 0,
                'end_line'    => $stmt->getEndLine()   ?? 0,
                'content'     => nodeContent($lines, $stmt),
                'node_type'   => 'method',
                'visibility'  => visibilityStr($stmt->flags),
                'is_static'   => $stmt->isStatic(),
                'is_abstract' => $stmt->isAbstract(),
            ];

            if ($parentFqn) {
                $rels[] = [
                    'type'   => 'MEMBER_OF',
                    'source' => $fqn,
                    'target' => $parentFqn,
                    'kind'   => 'method',
                ];
            }

            // Simple call detection: scan method body for static calls and $this->calls
            if ($stmt->stmts) {
                extractCalls($stmt->stmts, $fqn, $rels);
            }
            continue;
        }

        // ── Global functions ──────────────────────────────────────────────
        if ($stmt instanceof Stmt\Function_) {
            $funcName = $stmt->name->toString();
            $fqn      = $currentNamespace ? "$currentNamespace\\$funcName" : $funcName;

            $symbols[] = [
                'fqn'         => $fqn,
                'name'        => $funcName,
                'kind'        => 'function',
                'namespace'   => $currentNamespace,
                'start_line'  => $stmt->getStartLine() ?? 0,
                'end_line'    => $stmt->getEndLine()   ?? 0,
                'content'     => nodeContent($lines, $stmt),
                'node_type'   => 'function',
                'visibility'  => 'public',
                'is_static'   => false,
                'is_abstract' => false,
            ];
            continue;
        }

        // ── Properties ───────────────────────────────────────────────────
        if ($stmt instanceof Stmt\Property && $parentFqn) {
            foreach ($stmt->props as $prop) {
                $propName = $prop->name->toString();
                $fqn      = "$parentFqn::$$propName";
                $symbols[] = [
                    'fqn'         => $fqn,
                    'name'        => $propName,
                    'kind'        => 'property',
                    'namespace'   => $currentNamespace,
                    'start_line'  => $prop->getStartLine() ?? 0,
                    'end_line'    => $prop->getEndLine()   ?? 0,
                    'content'     => '',   // properties are small; skip content to save tokens
                    'node_type'   => 'property',
                    'visibility'  => visibilityStr($stmt->flags),
                    'is_static'   => (bool)($stmt->flags & Stmt\Class_::MODIFIER_STATIC),
                    'is_abstract' => false,
                ];
                $rels[] = [
                    'type'   => 'MEMBER_OF',
                    'source' => $fqn,
                    'target' => $parentFqn,
                    'kind'   => 'property',
                ];
            }
        }
    }
}

/**
 * Best-effort extraction of CALLS relationships from a method body.
 * Only captures the most common patterns to keep the analyzer fast.
 */
function extractCalls(array $stmts, string $callerFqn, array &$rels): void {
    $traverser = new NodeTraverser();
    $traverser->addVisitor(new class($callerFqn, $rels) extends NodeVisitorAbstract {
        public function __construct(
            private string $callerFqn,
            private array  &$rels
        ) {}

        public function enterNode(Node $node) {
            // ClassName::method() static calls
            if ($node instanceof Node\Expr\StaticCall
                && $node->class instanceof Node\Name
                && $node->name instanceof Node\Identifier) {
                $this->rels[] = [
                    'type'   => 'CALLS',
                    'source' => $this->callerFqn,
                    'target' => $node->class->toString() . '::' . $node->name->toString(),
                ];
            }
            // $this->method() instance calls
            if ($node instanceof Node\Expr\MethodCall
                && $node->var instanceof Node\Expr\Variable
                && $node->var->name === 'this'
                && $node->name instanceof Node\Identifier) {
                $this->rels[] = [
                    'type'   => 'CALLS',
                    'source' => $this->callerFqn,
                    'target' => 'self::' . $node->name->toString(),
                ];
            }
            return null;
        }
    });
    $traverser->traverse($stmts);
}

walkStmts($ast, $lines, '', '', $symbols, $rels);

// ── Output ───────────────────────────────────────────────────────────────────

outputResult($filePath, $symbols, $rels);

function outputResult(string $filePath, array $symbols, array $rels): void {
    $result = [
        'file_path'     => $filePath,
        'language'      => 'php',
        'symbols'       => $symbols,
        'relationships' => $rels,
        'raw_ast'       => null,
    ];
    echo json_encode($result, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
}
