require 'pycall'
require 'nokogiri'
require 'active_support/core_ext/hash'

class Model
    # +file+:: relative or absolute filepath to model.h5 file.
    def initialize(file, word_idx_file)
        @file = file
        @word_idx_file = word_idx_file
        PyCall.exec "#{File.read('./python/model.py')}"
        self
    end

    def retrain
        p 'Retraining...'
        # Read all closed issues from repository
        self
    end

    def predict()
        PyCall.exec "pred = Predictor()"
        # TODO:: Read puzzle xml filepath from PDD Options/Config
        puzzles = prepare_puzzles('../tmp/test.xml')
        # TODO:: Get closed issue text from appropriate source
        closed_issue_text = """
        Needs to enable this check and fix all issues. For now, this is out of the scope of this issue. This issue appeared after update TargetRubyVersion to 2.3
        need to implement this method. For now, it's just a task, that prints a simple Running pdd... message to user
        Change the implementation of this method to also work in Windows machines. Investigate the possibility of use a gem for this. After that, remove the skip of the test `test_ignores_binary_files` in `test_sources.rb`.
        """
        idx = 1
        output = PyCall.eval("pred.predict(\"#{p}\", #{puzzles})")
        output.each do |issue, score|
            puts "#{idx} #{issue[0..50]} #{score}\n"
            idx += 1
        end
        self
    end

    private def prepare_puzzles(f)
        if File.exist?(f)
            xml = Nokogiri::XML(File.read(f))
            puzzles = Hash.from_xml(xml.to_s)
            puzzles = puzzles['puzzles']['puzzle']
            puzzles_text = puzzles.map { |p| p['body'] }
            puzzles_text
        else
            throw 'File not found!'
        end
    end
end


# Test Class
model = Model.new('./data/model.h5', './data/word_to_ix.pickle')
model.predict()
