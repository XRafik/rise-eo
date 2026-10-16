# Reference — what good code is, and how to audit for it

**Status:** reference note for the 2026-10 refactor audit of `src/rise_eo/`, written 2026-10-09. It adds no rules.
The rules are in [`AGENTS.md`](../../AGENTS.md) ("Code conventions") and [`code-comments.md`](code-comments.md).
This note explains where they come from and turns them into questions an auditor can ask. If this note and a
rule disagree, the rule wins; report the disagreement. Words in quotation marks were read on the cited page
on 2026-10-09; everything else is paraphrase.

## 1. What a function is

**A mapping.** In mathematics a function pairs each argument with one value. Frege's "Function and
Concept" (1891) adds the point that matters here: the function is the part of an expression that is left
when the argument is taken out. In the Geach translation, "a function by itself must be called incomplete,
in need of supplementation, or unsaturated". `2·( )³ + ( )` is the function; `1` or `4` completes it. He
also notes that `x(x - 4)` and `x² - 4x` are different expressions with the same "value-range": the same
argument always gives the same value. That is the idea behind a behaviour-preserving refactor. Two pieces
of code can look different and still be the same function.

**A black box.** SICP §1.1.8 says "it is crucial that each procedure accomplishes an identifiable task that
can be used as a module in defining other procedures", and "A user should not need to know how the
procedure is implemented in order to use it." A function is good when a caller can use it from its name
and signature alone.

**A hidden decision.** Parnas (1972) compared two ways of splitting the same program into modules. He
recommends that "one begins with a list of difficult design decisions or design decisions which are likely
to change. Each module is then designed to hide such a decision from the others." So a module is judged by
what it hides, not by its size.

**Command or query.** Meyer's command-query separation, as Fowler states it: "Queries: Return a result and
do not change the observable state of the system (are free of side effects). Commands: Change the state of
a system but do not return a value." Fowler allows exceptions: "Popping a stack is a good example of a
query that modifies state."

**Pure or effectful.** A pure function's result depends only on its arguments, and it changes nothing
else. You can test it with a value, cache it, and reorder it. An effectful function writes files, calls
the network or reads a clock. You test it by its effects, and the order of calls matters.

**What this means in fsd.**
- Raster ops take and return `(data, profile)`, so they chain as `sequence=[(func, kwargs), ...]`
  (`AGENTS.md`). They are close to pure mappings. A small one is fine if it hides a real decision, such as
  how to resample to a reference grid. Judge it by Frege and SICP: does it map one well-defined input to
  one output, and can it be used without reading the body?
- The verbs (`rise_eo.download`, `rise_eo.create_training_data`, `rise_eo.run_inference`, `rise_eo.deploy`) are
  commands. Their real result is files on disk; they return a path or a handle to what they wrote. That
  is Fowler's "pop" case, and it is acceptable. Judge a verb by Parnas: which decision does it hide from the
  user (runners, storage backends, grid cells), and does a change to that decision stay inside it?
- **Size.** A pure op should be small, because its whole contract is in its signature. A command can be
  long if its interface stays simple and it hides a lot ("deep", §2). Do not count lines. Count what a
  caller must know.
- **Names.** Name a query by what it returns (`archive_catalog_filepath`). Name a command by what it does
  (`download`). A name that needs a comment to decode is a finding (§6, Q5).

## 2. Principles that matter for a behaviour-preserving audit

| Principle | Source | What it means here |
|---|---|---|
| Refactoring is "a change made to the internal structure of software to make it easier to understand and cheaper to modify without changing its observable behavior." | Fowler, bliki (2004); *Refactoring* 2nd ed. ch. 2 | The audit proposes only changes that keep outputs, files, errors and logs the same. Anything else is a behaviour change and needs its own PR title. |
| "A code smell is a surface indication that usually corresponds to a deeper problem", and smells do not *always* mean a problem. | Fowler, bliki (2006); *Refactoring* 2nd ed. ch. 3 | A smell is a reason to look, not a verdict. The smells that matter here are Duplicated Code, Long Function, Long Parameter List, Shotgun Surgery, Speculative Generality, Middle Man and Comments. The 2nd edition lists no "Dead Code" smell; it has the refactoring "Remove Dead Code" (p. 237). |
| Comments used "as a deodorant": "try first to refactor the code so that any comment would be superfluous." | Fowler & Beck, draft of *Refactoring* ch. 3 | A comment that explains confusing code points to the code to fix. A hazard comment is a different thing (§3). |
| Beck's rules of simple design, in Fowler's wording: passes the tests, reveals intention, no duplication, fewest elements. "The rules are in priority order". | Fowler, bliki "BeckDesignRules" (2015), pointing to Beck's "White Book" (*Extreme Programming Explained*, 1st ed., p. 57) | "Fewest elements" comes last. Line count is a measure, not a target. Never cut lines at the cost of the tests or of intent. |
| Complexity is "anything related to the structure of a system that makes it hard to work on the development of that system". Its causes are dependencies and obscurity. | Ousterhout, *A Philosophy of Software Design* ch. 2; CS190 notes | Ask two questions: what else must change when this changes (dependencies)? What must a reader already know (obscurity)? |
| Deep modules: "The best methods are those that provide a lot of functionality but have a very simple interface". Shallow ones hide little. | Ousterhout, ch. 4; aposd-vs-clean-code | A wrapper that adds a name but hides no decision is a cost. |
| Information leakage and pass-through variables. | Ousterhout, ch. 5 and §7.5 (titles from the TOC) | The same knowledge spread over several modules, or a value handed down through layers that do not use it. |
| "Whenever possible, define errors out of existence"; "reduce the number of places where exceptions must be handled". | Ousterhout, ch. 10; CS190 notes | Change the semantics so the error cannot happen. Do not catch and hide it. See §5 for the limit in fsd. |
| Simple means "one fold": not interleaved. "simple is actually an objective notion", while "easy is relative". | Hickey, "Simple Made Easy" (2011) | "I find this hard to read" is about *easy*. "These two concerns are braided together" is about *simple*, and you can check it. |
| Simplicity "keeps programs short and manageable"; clarity "makes sure they are easy to understand, for people as well as machines"; generality means programs "work well in a broad range of situations". | Kernighan & Pike, *The Practice of Programming*, preface | Generality is earned by real cases, not guessed. |
| Debugging is twice as hard as writing, so code written as cleverly as you can manage is code you cannot debug (paraphrased). | Kernighan & Plauger, *The Elements of Programming Style*, 2nd ed., ch. 2 (via Wikiquote) | Prefer the plain version. A clever one-liner in a data path is a finding. |
| "Errors should never pass silently. Unless explicitly silenced." "If the implementation is hard to explain, it's a bad idea." | PEP 20 | `raise_error=False` style flags are explicit silencing and allowed. An unlogged `except: pass` is not. |
| "duplication is far cheaper than the wrong abstraction"; "prefer duplication over the wrong abstraction". | Metz, "The Wrong Abstraction" (2016) | Two similar blocks are not automatically one function. Merge only when they change for the same reason. |

## 3. Is good code its own documentation?

**The case for yes.** *Clean Code* ch. 4, as quoted by Ousterhout from p. 54: "The proper use of comments
is to compensate for our failure to express ourselves in code. … Comments are always failures." In their
2024–25 discussion Martin adds "I'm not hostile to comments in general. I *am* very hostile to gratuitous
comments." Fowler's "deodorant" makes the same point more gently.

**The case for no.** Ousterhout, chs. 12–13 and his CS190 notes: "Comments should describe things that are
not obvious from the code", because "Code alone can't represent cleanly all the information in the mind of
the designer." He says comments carry intent, the reason for a choice, and the abstraction an interface
offers. He estimates he would write "5-10x more lines of comments" than Martin.

**The far end.** Knuth's "Literate Programming" (1984): "Instead of imagining that our main task is to
instruct a computer what to do, let us concentrate rather on explaining to human beings what we want a
computer to do." The program becomes an essay with code inside it.

**Under all three: Naur, "Programming as Theory Building" (1985).** Naur argues that programming is the
programmers building a theory, "any documentation being an auxiliary, secondary product." In his case study,
"the program text and its documentation has proved insufficient as a carrier of some of the most important
design ideas." Rebuilding the theory "merely from the documentation, is strictly impossible." And: "the very
notion of qualities such as simplicity and good structure can only be understood in terms of the theory of
the program."

This answers "everything is subjective". Readability is not a property of the text alone. It is a relation
between the text and the reader's theory of the program. Code that was clear to its author can be opaque to
the next person, or to the author a year later, without being any worse. So the honest goal is not "code
anyone can read cold". It is **code from which the theory can be recovered**. Names and structure carry most
of it. Hazard comments, specs, ADRs and tests carry the parts code cannot.

**Where the sources agree.** All of them want code that a reader can understand. Ousterhout's summary of the
discussion, which Martin called fair, says "We agree that implementation code only needs comments when the
code is nonobvious." Nobody defends
comments that restate the code.

**Where they truly disagree.** Whether an interface needs prose (Ousterhout yes, Martin mostly not for
internal code). Whether a long name beats a short name plus a comment. Whether a missing comment costs more
than a wrong one. These are judgement calls, and the sources do not settle them.

**fsd's position.** Code carries the *what* and the *how*, through names, signatures and structure. A
comment carries only what code cannot: the hazard, the invariant, the reason a tempting change is wrong.
That is `code-comments.md`'s rule, "Cut the changelog. Keep the hazard." It sits between Martin and
Ousterhout: Martin's suspicion of narrating comments, plus Ousterhout's view that some knowledge exists only
in prose. Naur explains why the hazard comment is the one to keep: it is the piece of the theory that the
next programmer is most likely to lack.

## 4. What is objective and what is judgement

**Objective:** two careful auditors would reach the same answer, often with a tool.
- Unused imports, variables, arguments, functions and options. `ruff check --select F401,F841,ARG src/rise_eo`
  finds the first three; the rest need `grep` for callers.
- Near-copies of code (read and compare; a diff of the two blocks settles it).
- A rule in `AGENTS.md` or `code-comments.md` that the code breaks (a bare "tile", I/O outside
  `rise_eo.storage`, a date-stamped comment, a changelog comment).
- Cyclomatic complexity, McCabe (1976): the number of independent paths through a function. McCabe shows
  that "complexity is independent of physical size" and used "10 which seems like a reasonable, but not
  magical, upper limit". He allowed a large case statement as an exception. `ruff check --select C901
  src/rise_eo` uses 10; on 2026-10-09 it reports 24 functions. A high number is a fact. Whether to split the
  function is judgement.

**Judgement:** a reasonable reader could disagree.
- Whether a name is clear, whether a function is too long, whether an abstraction is right, whether two
  near-copies should merge (Metz), whether a module is too shallow.

**Label every finding** `objective` or `judgement`. A judgement finding states its reasoning in one or two
sentences and names the source or rule it leans on, so the maintainer can disagree with the reasoning
rather than with a feeling.

## 5. Tensions with fsd's rules

- **Ousterhout's "define errors out of existence" vs `AGENTS.md` "Keep the safety checks".** They agree when
  an error can be removed by changing semantics without losing information. Example: deleting a file that is
  already gone can succeed. They conflict when "defining it away" would let a bad ROI, date or path through,
  or let a failed download leave a gap. Nodata is 0, so a gap can pass for real values. fsd leans hard toward
  the safety check. Ousterhout's own book has a section called "Taking it too far" (§10.10).
- **Clean Code's tiny functions vs Ousterhout's deep modules vs `AGENTS.md` "smallest code".** "Smallest
  code" means the least code overall, not the most functions. Splitting into many small functions often adds
  code and interfaces. Ousterhout's warning about entangled methods applies: if you must read both to
  understand one, they should be one. fsd leans toward Ousterhout: no speculative abstractions, and few
  layers.
- **DRY vs Metz.** Spec 102 A7 cites DRY for "reuse an fsd helper before you write". Metz warns that merging
  look-alike code into the wrong abstraction costs more than the copies. fsd's rules lean DRY for new code
  and say nothing about merging old code. In the audit, merge only when the copies change for the same reason.
- **"Few comments" vs Ousterhout's pro-comment stance.** `code-comments.md` sets a default of none, but keeps
  invariants, failure modes, deliberate choices, dependency traps and prohibitions. That is close to
  Ousterhout's list of what code cannot say, minus interface prose. fsd leans to fewer comments, with a firm
  exception for hazards.
- **Ousterhout's pass-through variables vs fsd's per-hop kwarg forwarding.** Ousterhout calls a value
  handed through layers that do not use it a red flag (§7.5). fsd forwards each verb kwarg explicitly on
  every runner on purpose. A default once hid a dropped `collection=` (CONTRIBUTING's review checklist), and
  each hop has a forwarding test. fsd's rule wins. Flag a pass-through only if it has no test.
- **Ousterhout's "shallow wrapper" vs Parnas's "hide a decision".** `storage/fs.py` looks shallow: thin
  functions over fsspec. But it hides the backend choice (local, Blob, S3), which is the decision most likely
  to change, and its traps (`code-comments.md` keeps it above the comment target for that reason). Parnas
  decides: a wrapper that hides a changeable decision is not shallow.

## 6. Audit rubric

Ask these of each module and function. Each question names its basis. Never propose removing a check that
guards user input or prevents silent data loss (`AGENTS.md`, "Keep the safety checks").

1. **[objective]** Is any function, parameter, option, branch or import unused? Check callers with `grep`,
   including tests, notebooks, Snakefiles and `docs/`. (Fowler "Remove Dead Code"; Beck rule 4; `AGENTS.md`
   "no speculative options")
2. **[objective]** Is there a near-copy of this block elsewhere in `src/rise_eo`? If yes, record both places.
   Then, as **[judgement]**, say whether they change for the same reason, before you propose a merge.
   (Fowler "Duplicated Code"; Metz)
3. **[objective]** Does the code break a written rule: a bare "tile", file I/O outside `rise_eo.storage`, a
   changelog or date-stamped comment, more than one spec reference per function, a docstring past
   `code-comments.md`'s ~15 lines? (`AGENTS.md` "Code conventions"; `code-comments.md`)
4. **[objective]** What is the function's cyclomatic complexity (`ruff check --select C901`)? Above 10,
   say what the independent paths are. Splitting is a separate judgement. (McCabe 1976)
5. **[judgement]** Does the name say what the function returns (query) or does (command)? Would a reader
   misuse it from the name and signature alone, or need a comment to decode it? (§1; SICP §1.1.8;
   Ousterhout ch. 14)
6. **[judgement]** Does the function do more than one thing that a caller could name and want separately?
   If you split it, would the parts be entangled, so that one cannot be read without the other? (Ousterhout,
   deep vs shallow and entanglement; Fowler "Long Function")
7. **[judgement]** Is this a shallow wrapper: an extra interface that hides no decision? Before you call it
   shallow, name the decision it might hide (backend, runner, format version). If it hides one, keep it.
   (Ousterhout ch. 4; Parnas 1972; Fowler "Middle Man")
8. **[objective]** Is a parameter passed through several hops without being used? Record the chain.
   **Flag, don't fix:** per-hop kwarg forwarding in the verbs and runners is deliberate and protected by
   CONTRIBUTING's review checklist. Report only a hop that has no forwarding test. (Ousterhout §7.5;
   `CONTRIBUTING.md`)
9. **[judgement]** Is there an option, hook or abstraction with only one real use, added for a future that
   has not come? (Fowler "Speculative Generality"; Naur on flexibility, "a program feature whose usefulness
   depends entirely on future events"; `AGENTS.md`)
10. **[objective]** Does a comment restate the code, narrate history, or point to a spec more than once? Each
    is a cut under `code-comments.md`. **[judgement]** Conversely: is there a trap here (an invariant, a
    failure mode, a dependency quirk) that has no hazard comment and that the next reader would get wrong?
    (`code-comments.md`; Ousterhout ch. 13; Naur)
11. **[objective]** Does any `except` swallow an error: catch broadly, then return a default or continue,
    without logging or re-raising? Flag it. Never propose removing a check on user input or one that stops
    silent data loss. Note whether the silencing is explicit and opt-in (like `raise_error=False`). (PEP 20;
    Shore "Fail Fast" via spec 102 A7; `AGENTS.md` "Keep the safety checks")
12. **[objective]** Is there a leftover legacy workaround: a compatibility shim for an old archive layout, a
    branch for a case the code no longer produces, or code copied from `fetch_satdata/` that fsd's own
    structure makes unnecessary? (`AGENTS.md` "No compatibility shims"; Fowler "Remove Dead Code")
13. **[judgement]** Does one change in behaviour need edits in many files (shotgun surgery), or does one
    file change for many unrelated reasons? (Fowler "Shotgun Surgery", "Divergent Change"; Ousterhout,
    dependencies)
14. **[judgement]** Is any code clever where it could be plain: dense comprehensions or index tricks in a
    data path, whose bugs would be costly to find? (Kernighan & Plauger; PEP 20 "Readability counts")
15. **[objective]** For every proposed change: does it keep observable behaviour (outputs, files, errors,
    log lines) the same? If not, it is out of scope for this audit. Record it as a separate finding.
    (Fowler's definition of refactoring)

## 7. Sources

Each entry says what it contributed. "Read" means the page was opened on 2026-10-09.

- **G. Frege, "Function and Concept" (1891)**, translated by P. Geach. Read in a scanned reprint,
  <https://fitelson.org/proseminar/frege_fac.pdf>. Its editorial notes suggest it is M. Beaney (ed.),
  *The Frege Reader* (Blackwell, 1997); I did not confirm the volume. Contributed: the "unsaturated"
  quotation and the same-value-range example (§1).
- **H. Abelson, G. J. Sussman, J. Sussman, *Structure and Interpretation of Computer Programs*, 2nd ed.,
  §1.1.8 "Procedures as Black-Box Abstractions"**, MIT Press full text,
  <https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-10.html>.
  Contributed: the two black-box quotations (§1).
- **D. L. Parnas, "On the Criteria To Be Used in Decomposing Systems into Modules", *Communications of the
  ACM* 15(12), 1972, pp. 1053–1058.** Read at
  <https://teaching.csse.uwa.edu.au/units/CITS4401/readings/parnasmodules.htm>. Contributed: the
  information-hiding quotation (§1) and the test that decides "shallow wrapper" findings (§5, Q7).
- **M. Fowler, "CommandQuerySeparation" (2005)**, <https://martinfowler.com/bliki/CommandQuerySeparation.html>.
  Contributed: the CQS definition and the "pop" exception (§1). It attributes CQS to B. Meyer, *Object-Oriented
  Software Construction*; I did not read Meyer.
- **M. Fowler, "DefinitionOfRefactoring" (2004)**, <https://martinfowler.com/bliki/DefinitionOfRefactoring.html>.
  Contributed: the definition of refactoring (§2, Q15).
- **M. Fowler, "CodeSmell" (2006)**, <https://martinfowler.com/bliki/CodeSmell.html>. Contributed: the
  definition of a smell and the point that smells are not verdicts (§2).
- **M. Fowler, *Refactoring*, 2nd ed. (Addison-Wesley, 2018), table of contents**, Pearson sample,
  <https://www.pearson.de/media/muster/toc/toc_9780134757698.pdf>. Contributed: the ch. 3 smell names, and
  the fact that "Dead Code" is a refactoring (p. 237), not a smell heading (§2, rubric). Book text not read.
- **M. Fowler, K. Beck, "Bad Smells in Code", a draft of *Refactoring* ch. 3**,
  <https://www.laputan.org/pub/patterns/fowler/smells.pdf>. A pre-publication draft; the published wording may
  differ. Contributed: the "deodorant" and "superfluous" quotations and the Middle Man description (§2, Q7).
- **M. Fowler, "BeckDesignRules" (2015)**, <https://martinfowler.com/bliki/BeckDesignRules.html>.
  Contributed: the four rules in Fowler's wording and their priority order (§2). For Beck's own wording it
  points to "the first edition of The White Book (p 57)", i.e. *Extreme Programming Explained*; I did not
  read Beck.
- **J. Ousterhout, *A Philosophy of Software Design* (Yaknyam Press), table of contents**,
  <https://external.dandelon.com/download/attachments/dandelon/ids/DE0012CB5DBB06A590C33C12582B900391C1B.pdf>.
  Contributed: the chapter and section numbers cited (chs. 2, 4, 5, 7.5, 10, 10.10, 12, 13, 14). Book text not
  read.
- **J. Ousterhout, CS190 lecture notes (Stanford, winter 2018)**: "complexity"
  <https://web.stanford.edu/~ouster/cgi-bin/cs190-winter18/lecture.php?topic=complexity>, "exceptions"
  <https://web.stanford.edu/~ouster/cgi-bin/cs190-winter18/lecture.php?topic=exceptions>, "comments"
  <https://web.stanford.edu/~ouster/cgi-bin/cs190-winter18/lecture.php?topic=comments>. The author's own
  notes. Contributed: the complexity definition, dependencies and obscurity, "define errors out of
  existence", and the comment quotations (§2, §3).
- **J. Ousterhout and R. C. Martin, "aposd-vs-clean-code" (discussion, 2024-09 to 2025-02)**,
  <https://github.com/johnousterhout/aposd-vs-clean-code>. Both authors' own words. Contributed: the deep
  and entangled definitions, the quotation from *Clean Code* p. 54, Martin's "gratuitous comments" line, the
  "5-10x" estimate, and the agreed sentence on nonobvious code (§2, §3, §5). I did not read *Clean Code*
  itself.
- **R. Hickey, "Simple Made Easy", Strange Loop 2011.** Read in a community transcript,
  <https://github.com/matthiasn/talk-transcripts/blob/master/Hickey_Rich/SimpleMadeEasy.md>, not the video.
  Contributed: simple vs easy and "complect" (§2).
- **B. W. Kernighan, R. Pike, *The Practice of Programming* (1999), preface**,
  <https://www.cs.princeton.edu/~bwk/tpop.webpage/preface.html>. Contributed: simplicity, clarity and
  generality (§2).
- **B. W. Kernighan, P. J. Plauger, *The Elements of Programming Style*, 2nd ed. (1978), ch. 2.** Not read.
  The attribution is from <https://en.wikiquote.org/wiki/Brian_Kernighan>, which is why the debugging line is
  paraphrased (§2, Q14).
- **PEP 20, "The Zen of Python"**, <https://peps.python.org/pep-0020/>. Contributed: silent errors,
  readability, hard-to-explain implementations (§2, Q11, Q14).
- **S. Metz, "The Wrong Abstraction" (2016-01-20)**, <https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction>.
  Contributed: the counterweight to DRY (§2, §5, Q2).
- **D. E. Knuth, "Literate Programming", *The Computer Journal* (1984).** Read in the author's submitted
  version, <http://www.literateprogramming.com/knuthweb.pdf>. Contributed: the "explaining to human beings"
  quotation (§3).
- **P. Naur, "Programming as Theory Building", *Microprocessing and Microprogramming* (1985).** Read in the
  reprint in A. Cockburn, *Agile Software Development* (2002), pp. 227–235, at
  <https://pages.cs.wisc.edu/~remzi/Naur.pdf>. Contributed: the theory-building view, the limits of
  documentation, quality relative to the theory, and the cost of speculative flexibility (§3, Q9, Q10).
- **T. J. McCabe, "A Complexity Measure", *IEEE Transactions on Software Engineering* SE-2(4), 1976,
  pp. 308–320**, <http://literateprogramming.com/mccabe.pdf>. Contributed: cyclomatic complexity, its
  independence from size, and the "not magical" limit of 10 (§4, Q4). I read pp. 308–311 and 314.
- **Local:** [`AGENTS.md`](../../AGENTS.md) "Code conventions"; [`code-comments.md`](code-comments.md);
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md) "Review checklist";
  [`specs/102-contributor-readiness.md`](../../specs/102-contributor-readiness.md) Amendment A7 (DRY, Shore
  "Fail Fast", PEP 20 for the safety-check rule).

**Searched.** Web searches: Parnas 1972 criteria paper; Knuth "Literate Programming" 1984 quotation;
Kernighan's debugging quotation and *The Elements of Programming Style*; Kernighan & Pike preface; Hickey
"Simple Made Easy" transcript; McCabe 1976; Frege "Function and Concept" "unsaturated"; the Ousterhout table
of contents; Ousterhout "dependencies and obscurity" on stanford.edu; the *Refactoring* 2nd ed. ch. 3 smell
list. **Could not access:** the full text of *Clean Code*, *A Philosophy of Software Design*, *Refactoring*
(published edition), *Object-Oriented Software Construction*, *Extreme Programming Explained* and *The
Elements of Programming Style*, and the Hickey video. Claims from these rest on the pages listed above.
Quotations fetched through a summariser were re-checked against the raw page text.
